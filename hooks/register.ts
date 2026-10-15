import { atom, update } from 'claude-code'
import type { EngineInterface, PluginOptions, Register, ToolCallInput, ToolCallResult } from 'claude-code'

import { COMMAND_USAGE, commandArgv, formatCount, parseSaving, savingLine, utf8Bytes } from './savings'

// The lowest minimum_tokens the engine accepts (engine/hook.py config()).
const FLOOR_TOKENS = 1000
// The engine's own ceiling for one captured output.
const MAX_BYTES = 4 * 1024 * 1024
// Covers the engine's 300 s helper deadline plus process start-up.
const PROCESS_TIMEOUT_MS = 330_000
const SETUP_TIMEOUT_MS = 600_000

const total = atom({ plugin: 'context-saver', key: 'session' } as const, { outputs: 0, tokens: 0 })

type Dollar = EngineInterface

/** The /config options as the engine reads them (engine/hook.py config()). */
function engineEnv(options: PluginOptions): Record<string, string> {
  return {
    CONTEXT_SAVER_MODEL: String(options.model ?? 'sonnet'),
    CONTEXT_SAVER_EFFORT: String(options.effort ?? 'medium'),
    CONTEXT_SAVER_SUMMARIZE_SHELL: options.summarize_shell === false ? '0' : '1',
  }
}

/** Runs the engine CLI (bin/context-saver) with the plugin's options in its environment. */
function runEngine($: Dollar, env: Record<string, string>, args: string[], init: { stdin?: string; timeoutMs?: number } = {}) {
  // Through sh, so an install that drops the file's exec bit still works.
  return $.process.run(['sh', `${$.plugin.root}/bin/context-saver`, ...args], { env, timeoutMs: PROCESS_TIMEOUT_MS, ...init })
}

// One setup check per module load, shared by session.start and the first tool
// call (a plugin installed mid-session may never see session.start).
let readiness: Promise<boolean> | undefined
// Output under the engine's minimum_tokens in UTF-8 bytes can never qualify
// (tokens <= bytes), so it skips the Python round trip.
let minimumBytes = FLOOR_TOKENS

/** The engine's minimum_tokens from config.json; the floor when it cannot be read. */
async function engineMinimum($: Dollar, env: Record<string, string>): Promise<number> {
  const limits = await runEngine($, env, ['limits'])
  try {
    const minimum = Number(JSON.parse(limits.stdout).minimum_tokens)
    return limits.exitCode === 0 && Number.isFinite(minimum) ? minimum : FLOOR_TOKENS
  } catch {
    return FLOOR_TOKENS
  }
}

/** Builds or checks the Python environment; false leaves output uncompressed. */
async function ensureEngine($: Dollar, env: Record<string, string>): Promise<boolean> {
  try {
    const setup = await runEngine($, env, ['ensure'], { timeoutMs: SETUP_TIMEOUT_MS })
    if (setup.exitCode !== 0) {
      $.ui.log(`context-saver: Python setup failed, output stays uncompressed (${setup.stderr.trim().slice(-300)}); retry with /context-saver setup`)
      return false
    }
    if (setup.stdout.trim()) $.ui.log(setup.stdout.trim())
    minimumBytes = await engineMinimum($, env)
    return true
  } catch {
    // Interrupted or timed out: try again on the next call.
    readiness = undefined
    return false
  }
}

function engineReady($: Dollar, env: Record<string, string>): Promise<boolean> {
  readiness ??= ensureEngine($, env)
  return readiness
}

/** The tool's own arguments, without the envelope keys the engine adds. */
function toolInput(e: ToolCallInput): Record<string, unknown> {
  const { tool: _tool, tool_use_id: _id, agentId: _agent, requestMeta: _meta, ...input } = e as ToolCallInput & Record<string, unknown>
  return input
}

type Captured = { output: string; response: unknown; isPersisted: boolean }

/** The complete output the engine should judge, or undefined to leave the call alone. */
async function capturedOutput($: Dollar, e: ToolCallInput, ran: ToolCallResult): Promise<Captured | undefined> {
  if (ran.deny !== undefined || ran.isError) return undefined
  if (e.tool === 'Bash') {
    const result = ran.result as Record<string, unknown> | undefined
    if (!result || result.interrupted || result.isImage || result.backgroundTaskId) return undefined
    let stdout = result.stdout
    const isPersisted = typeof result.persistedOutputPath === 'string'
    if (typeof result.persistedOutputPath === 'string') {
      // Large output reaches the model as a preview; the full text is on disk.
      // The summary takes the preview's place (the savings line says so).
      if (Number(result.persistedOutputSize ?? 0) > MAX_BYTES) return undefined
      stdout = await $.fs.read(result.persistedOutputPath)
    }
    const output = [stdout, result.stderr].filter(part => typeof part === 'string' && part).join('\n')
    return { output, response: { output, exit_code: 0 }, isPersisted }
  }
  if (String(e.tool).startsWith('mcp__') && Array.isArray(ran.result)) {
    return { output: ran.text ?? '', response: { content: ran.result }, isPersisted: false }
  }
  return undefined
}

export const register: Register = (on, options) => {
  const env = engineEnv(options)

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'context-saver',
      description: 'Context saver: token savings, status, setup, enable or disable',
      argumentHint: 'stats [days] | status | setup | enable | disable | review <note>',
    })
    if (await engineReady($, env)) {
      // Prunes old archives; once a week it also reports the savings.
      const weekly = await runEngine($, env, ['session-start'])
      const context = weekly.exitCode === 0 && weekly.stdout.trim()
        ? JSON.parse(weekly.stdout)?.hookSpecificOutput?.additionalContext : undefined
      if (typeof context === 'string') $.ui.log(`context-saver: ${context}`)
    }
    return next(e)
  })

  on('tool.call', async ($, e, next) => {
    const ran = await next(e)
    const captured = await capturedOutput($, e, ran)
    if (!captured || !(await engineReady($, env))) return ran
    if (captured.output.length < minimumBytes && utf8Bytes(captured.output) < minimumBytes) return ran
    const event = {
      hook_event_name: 'PostToolUse', tool_name: e.tool, tool_input: toolInput(e),
      tool_use_id: e.tool_use_id, session_id: await $.session.id(), cwd: await $.session.cwd(),
      tool_response: captured.response,
    }
    const done = await runEngine($, env, ['process'], { stdin: JSON.stringify(event) })
    if (done.exitCode !== 0 || !done.stdout.trim()) return ran
    const replacement = JSON.parse(done.stdout)?.stopReason
    const saving = typeof replacement === 'string' ? parseSaving(replacement) : undefined
    if (!saving) return ran

    const sum = await update($, total, value => ({ outputs: value.outputs + 1, tokens: value.tokens + saving.saved }))
    const line = savingLine(saving, sum.tokens, captured.isPersisted)
    $.ui.log(line)
    $.ui.status(`⛁ ${formatCount(sum.tokens)} tokens saved · ${sum.outputs} output${sum.outputs === 1 ? '' : 's'}`)
    if (options.toast === true) $.ui.toast(line)

    // Context added by hooks beneath stays with the result.
    const context = ran.context === undefined ? {} : { context: ran.context }
    if (e.tool === 'Bash') {
      // The persisted file still holds the original; the model gets the summary.
      return { result: { ...(ran.result as object), stdout: replacement, stderr: '', persistedOutputPath: undefined, persistedOutputSize: undefined }, ...context } as ToolCallResult
    }
    return { result: [{ type: 'text', text: replacement }], ...context } as ToolCallResult
  }).catch(($, e, next) => next(e))

  on('command.run', { command: 'context-saver' }, async ($, e) => {
    const argv = commandArgv(e.args)
    if (!argv) return { text: COMMAND_USAGE }
    const ran = await runEngine($, env, argv, { timeoutMs: SETUP_TIMEOUT_MS })
    if (argv[0] === 'setup') readiness = Promise.resolve(ran.exitCode === 0)
    const output = (ran.stdout.trim() || ran.stderr.trim()).slice(0, 20_000)
    return { text: ran.exitCode === 0 ? output : `context-saver ${argv[0]} failed (exit ${ran.exitCode}): ${output}` }
  })
}
