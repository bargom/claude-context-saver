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
// `ensure --background` only checks the venv or starts the setup detached.
const ENSURE_TIMEOUT_MS = 30_000
const DELIVERED_TIMEOUT_MS = 10_000
// bin/context-saver exits 75 while the setup runs in the background.
const SETUP_RUNNING = 75
// When the next check may run: while the setup runs, and after it failed.
const RUNNING_RETRY_MS = 15_000
const FAILED_RETRY_MS = 10 * 60_000

const total = atom({ plugin: 'context-saver', key: 'session' } as const, { outputs: 0, tokens: 0 })

type Dollar = EngineInterface

/** The /config options as the engine reads them (engine/hook.py config()). */
function engineEnv(options: PluginOptions): Record<string, string> {
  return {
    CONTEXT_SAVER_EFFORT: String(options.effort ?? 'medium'),
    CONTEXT_SAVER_SUMMARIZE_SHELL: options.summarize_shell === false ? '0' : '1',
  }
}

/** Runs the engine CLI (bin/context-saver) with the plugin's options in its environment. */
function runEngine($: Dollar, env: Record<string, string>, args: string[], init: { stdin?: string; timeoutMs?: number } = {}) {
  // Through sh, so an install that drops the file's exec bit still works.
  return $.process.run(['sh', `${$.plugin.root}/bin/context-saver`, ...args], { env, timeoutMs: PROCESS_TIMEOUT_MS, ...init })
}

/** The JSON object an engine action printed, or undefined when it printed none. */
function engineReply(stdout: string): Record<string, unknown> | undefined {
  try {
    const reply: unknown = JSON.parse(stdout)
    return reply !== null && typeof reply === 'object' ? reply as Record<string, unknown> : undefined
  } catch {
    return undefined
  }
}

const configErrorLine = (message: string) => `context-saver: config.json has an error (${message})`
const setupFailedLine = (detail: string) =>
  `context-saver: Python setup failed, output stays uncompressed (${detail.trim().slice(-300)}); retry with /context-saver setup`

/**
 * Whether the Python engine can run, for one plugin load. Never waits for a
 * setup: `ensure --background` starts it detached and exits 75 while it runs,
 * and output stays uncompressed until a later check finds the venv ready.
 */
type EngineState = {
  env: Record<string, string>
  isReady: boolean
  // No check before this time ($.clock milliseconds).
  retryAt: number
  // The one check in flight, shared by every caller.
  checking?: Promise<boolean>
  // Output under the engine's minimum_tokens in UTF-8 bytes can never qualify
  // (tokens <= bytes), so it skips the Python round trip.
  minimumBytes: number
  // Notices this plugin load has logged.
  logged: Set<string>
}

/** Logs a notice the first time this plugin load sees it. */
function logOnce($: Dollar, engine: EngineState, line: string) {
  if (engine.logged.has(line)) return
  engine.logged.add(line)
  $.ui.log(line)
}

/** The engine's minimum_tokens from config.json; the floor when it cannot be read. */
async function refreshMinimum($: Dollar, engine: EngineState) {
  engine.minimumBytes = FLOOR_TOKENS
  try {
    const limits = await runEngine($, engine.env, ['limits'])
    const reply = engineReply(limits.stdout)
    // A broken config.json answers configError with minimum_tokens: null.
    if (typeof reply?.configError === 'string') return logOnce($, engine, configErrorLine(reply.configError))
    const minimum = reply?.minimum_tokens
    if (limits.exitCode === 0 && typeof minimum === 'number' && Number.isFinite(minimum) && minimum > 0) engine.minimumBytes = minimum
  } catch {
    // Keeps the floor; the engine applies its own minimum as well.
  }
}

/** One `ensure --background` unless a retry window is open: 0 ready, 75 setup running, else failed. */
async function ensureEngine($: Dollar, engine: EngineState): Promise<boolean> {
  const now = await $.clock.now()
  if (now < engine.retryAt) return false
  try {
    const ensured = await runEngine($, engine.env, ['ensure', '--background'], { timeoutMs: ENSURE_TIMEOUT_MS })
    if (ensured.exitCode === 0) {
      engine.isReady = true
      await refreshMinimum($, engine)
      return true
    }
    if (ensured.exitCode === SETUP_RUNNING) {
      engine.retryAt = now + RUNNING_RETRY_MS
      if (ensured.stdout.trim()) logOnce($, engine, ensured.stdout.trim())
      return false
    }
    engine.retryAt = now + FAILED_RETRY_MS
    $.ui.log(setupFailedLine(ensured.stderr))
  } catch (error) {
    // Could not start, or hung past the timeout: back off as for a failure.
    engine.retryAt = now + FAILED_RETRY_MS
    $.ui.log(setupFailedLine(String(error)))
  }
  return false
}

/** True once the engine can run; false while a setup runs or after one failed. */
function engineReady($: Dollar, engine: EngineState): Promise<boolean> {
  if (engine.isReady) return Promise.resolve(true)
  engine.checking ??= ensureEngine($, engine).finally(() => { engine.checking = undefined })
  return engine.checking
}

/** After `/context-saver setup`: ready when it worked, else the next call checks again. */
async function setupFinished($: Dollar, engine: EngineState, isSuccess: boolean) {
  engine.isReady = isSuccess
  engine.retryAt = 0
  if (isSuccess) await refreshMinimum($, engine)
}

/** The tool's own arguments, without the envelope keys the engine adds. */
function toolInput(e: ToolCallInput): Record<string, unknown> {
  // consent holds the person's own words for a permission press: never the engine's.
  const { tool: _tool, tool_use_id: _id, agentId: _agent, requestMeta: _meta, consent: _consent, ...input } =
    e as ToolCallInput & Record<string, unknown>
  return input
}

/** The token the engine logs with a candidate saving; `delivered` confirms it (engine common.DELIVERY_TOKEN). */
function deliveryToken(toolUseId: string | undefined): string {
  const id = (toolUseId ?? '').replace(/[^A-Za-z0-9_-]/g, '')
  return id ? `cs-${id}`.slice(0, 100) : `cs-${Date.now()}`
}

type Captured = { output: string; response: unknown; isPersisted: boolean }

// Claude Code's record of an MCP result over its token limit: the full result
// is in a file under the session's tool-results folder.
const SAVED_MCP_RESULT = /^Error: result \([^)]*\) exceeds maximum allowed tokens\. Output has been saved to (.+?)\.(?:\n|$)/

/** The file Claude Code saved an oversized MCP result to, or undefined. */
function savedResultPath(ran: ToolCallResult): string | undefined {
  // The host's own record only: text blocks an MCP server wrote never count.
  const record = typeof ran.result === 'string' ? ran.result : Array.isArray(ran.result) ? undefined : ran.text
  const path = typeof record === 'string' ? SAVED_MCP_RESULT.exec(record)?.[1] : undefined
  if (!path?.startsWith('/')) return undefined
  const parts = path.split('/')
  return parts.at(-2) === 'tool-results' && parts.at(-1) && !parts.includes('..') ? path : undefined
}

/** A file Claude Code saved, or undefined when it cannot be read (gone, or over 4 MiB). */
async function readSaved($: Dollar, path: string): Promise<string | undefined> {
  try {
    return await $.fs.read(path)
  } catch {
    return undefined
  }
}

/** The text blocks of an MCP result, joined as engine common.text_blocks joins them. */
function textOf(blocks: readonly unknown[]): string {
  return blocks.flatMap(block => {
    const { type, text } = (block ?? {}) as { type?: unknown; text?: unknown }
    return type === 'text' && typeof text === 'string' ? [text] : []
  }).join('\n')
}

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
      stdout = await readSaved($, result.persistedOutputPath)
      if (stdout === undefined) return undefined
    }
    const stderr = typeof result.stderr === 'string' ? result.stderr : ''
    const output = [stdout, stderr].filter(part => typeof part === 'string' && part).join('\n')
    // Claude Code reports no exit code, only its reading of a special one.
    const meaning = typeof result.returnCodeInterpretation === 'string'
      ? { return_code_interpretation: result.returnCodeInterpretation } : {}
    return { output, response: { output, stderr, ...meaning }, isPersisted }
  }
  if (!String(e.tool).startsWith('mcp__')) return undefined
  const savedPath = savedResultPath(ran)
  if (savedPath !== undefined) {
    // The model saw only the notice; the summary takes its place.
    const text = await readSaved($, savedPath)
    return text === undefined ? undefined : { output: text, response: { content: [{ type: 'text', text }] }, isPersisted: true }
  }
  if (!Array.isArray(ran.result)) return undefined
  const output = typeof ran.text === 'string' ? ran.text : textOf(ran.result)
  return { output, response: { content: ran.result }, isPersisted: false }
}

export const register: Register = (on, options) => {
  const env = engineEnv(options)
  // One readiness per plugin load, shared by session.start, tool calls and
  // /context-saver setup (a plugin installed mid-session may never see session.start).
  const engine: EngineState = { env, isReady: false, retryAt: 0, minimumBytes: FLOOR_TOKENS, logged: new Set() }

  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'context-saver',
      description: 'Context saver: token savings, status, setup, enable or disable',
      argumentHint: 'stats [days] | status | setup | enable | disable | review <note>',
    })
    if (await engineReady($, engine)) {
      // Prunes old archives; once a week it also reports the savings.
      const started = await runEngine($, env, ['session-start'])
      const reply = started.exitCode === 0 ? engineReply(started.stdout) : undefined
      if (typeof reply?.configError === 'string') logOnce($, engine, configErrorLine(reply.configError))
      const context = (reply?.hookSpecificOutput as { additionalContext?: unknown } | undefined)?.additionalContext
      if (typeof context === 'string') $.ui.log(`context-saver: ${context}`)
    }
    return next(e)
  })

  on('tool.call', async ($, e, next) => {
    const ran = await next(e)
    const captured = await capturedOutput($, e, ran)
    if (!captured) return ran
    const { minimumBytes } = engine
    if (captured.output.length < minimumBytes && utf8Bytes(captured.output) < minimumBytes) return ran
    // Never waits for a setup: until the engine is ready, output stays as it is.
    if (!(await engineReady($, engine))) return ran
    const token = deliveryToken(e.tool_use_id)
    const event = {
      hook_event_name: 'PostToolUse', tool_name: e.tool, tool_input: toolInput(e),
      tool_use_id: e.tool_use_id, session_id: await $.session.id(), cwd: await $.session.cwd(),
      tool_response: captured.response,
      // From a file Claude Code saved: the model would have seen only a preview or notice.
      ...(captured.isPersisted ? { persisted_output: true } : {}),
      delivery_token: token,
    }
    const done = await runEngine($, env, ['process'], { stdin: JSON.stringify(event) })
    if (done.exitCode !== 0 || !done.stdout.trim()) return ran
    const replacement = engineReply(done.stdout)?.stopReason
    const saving = typeof replacement === 'string' ? parseSaving(replacement) : undefined
    if (!saving) return ran
    // The ledger counts the saving once the replacement goes back; a failure only loses the count.
    await runEngine($, env, ['delivered', '--delivery-token', token], { timeoutMs: DELIVERED_TIMEOUT_MS }).catch(() => undefined)

    const sum = await update($, total, value => ({ outputs: value.outputs + 1, tokens: value.tokens + saving.saved }))
    const line = savingLine(saving, sum.tokens, captured.isPersisted)
    $.ui.log(line)
    $.ui.status(`⛁ ${formatCount(sum.tokens)} tokens saved · ${sum.outputs} output${sum.outputs === 1 ? '' : 's'}`)
    if (options.toast === true) $.ui.toast(line)

    // Context added by hooks beneath stays with the result.
    const context = ran.context === undefined ? {} : { context: ran.context }
    if (e.tool === 'Bash') {
      // The persisted file still holds the original; the model gets the summary.
      // structuredContent goes too: the Bash mapper would show it before stdout.
      return {
        result: { ...(ran.result as object), stdout: replacement, stderr: '', structuredContent: undefined, persistedOutputPath: undefined, persistedOutputSize: undefined },
        ...context,
      } as ToolCallResult
    }
    return { result: [{ type: 'text', text: replacement }], ...context } as ToolCallResult
  }).catch(($, e, next) => next(e))

  on('command.run', { command: 'context-saver' }, async ($, e) => {
    const argv = commandArgv(e.args)
    if (!argv) return { text: COMMAND_USAGE }
    const ran = await runEngine($, env, argv, { timeoutMs: SETUP_TIMEOUT_MS })
    if (argv[0] === 'setup') await setupFinished($, engine, ran.exitCode === 0)
    const output = (ran.stdout.trim() || ran.stderr.trim()).slice(0, 20_000)
    return { text: ran.exitCode === 0 ? output : `context-saver ${argv[0]} failed (exit ${ran.exitCode}): ${output}` }
  })
}
