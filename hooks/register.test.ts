import { describe, expect, test } from 'claude-code/testing'
import type { On, ProcessRunResult } from 'claude-code'

import { commandArgv, parseSaving, utf8Bytes } from './savings'

const BIG = 'src/a.js:12: function handler() { return 1; }\n'.repeat(300)
const SUMMARY = '[4200 tokens saved by Claude (sonnet)] (estimated context tokens)\n' +
  '[Context summary via sonnet: search; exit_code=0; estimated original tokens=5000]\n- handler is defined in src/a.js:12'

type EngineCall = { args: string[]; stdin?: string; env?: Record<string, string> }

const ran = (stdout = '', exitCode = 0): ProcessRunResult =>
  ({ exitCode, stdout, stderr: '', isStdoutTruncated: false, isStderrTruncated: false })

/** Stands for the engine CLI, the session and the UI beneath the plugin. */
function world(on: On, engineReply: (args: string[]) => ProcessRunResult = () => ran(), minimumTokens = 3500) {
  const calls: EngineCall[] = []
  const logs: string[] = []
  on('process.run', ($, e) => {
    // argv: sh, the launcher, then the engine CLI's own arguments.
    const args = e.argv.slice(2)
    calls.push({ args, stdin: e.init?.stdin, env: e.init?.env })
    if (args[0] === 'limits') return { value: ran(JSON.stringify({ minimum_tokens: minimumTokens })) }
    return { value: args[0] === 'process' || args[0] === 'stats' ? engineReply(args) : ran() }
  })
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('session.id', () => ({ value: 'session-1' }))
  on('session.cwd', () => ({ value: '/work' }))
  on('ui.log', ($, e) => { logs.push(e.text); return { value: undefined } })
  on('fs.read', () => ({ value: BIG }))
  on('ui.status', () => ({ value: undefined }))
  on('ui.toast', () => ({ value: undefined }))
  return { calls, logs }
}

const start = { cwd: '/work', surface: null, isInteractive: false } as const
const typed = (args: string) => ({ command: 'context-saver', args, origin: { kind: 'composer' as const },
  presentation: { isFullscreen: false, columns: 120 } })

describe('tool output compression', () => {
  test('large Bash output is replaced by the engine summary and the saving is shown', async ($, on) => {
    const { calls, logs } = world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })))
    on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: BIG, stderr: '', interrupted: false }, text: BIG }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n handler src' })

    expect((result.result as { stdout: string }).stdout).toBe(SUMMARY)
    const engineCall = calls.find(call => call.args[0] === 'process')
    const event = JSON.parse(engineCall?.stdin ?? '{}')
    expect(event.tool_name).toBe('Bash')
    expect(event.tool_input).toEqual({ command: 'rg -n handler src' })
    expect(event.tool_response).toEqual({ output: BIG, exit_code: 0 })
    expect(logs.some(line => line.includes('4,200 tokens saved (5,000 → 800) by Claude (sonnet) · session 4,200'))).toBe(true)
  })

  test('output Claude Code saved to a file is summarized from the file and counted against it', async ($, on) => {
    const { calls, logs } = world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })))
    const preview = '<persisted-output>\nOutput too large (14KB). Full output saved to: /tmp/out.txt\n</persisted-output>'
    on('tool.call', { tool: 'Bash' }, () => ({
      result: { stdout: preview, stderr: '', interrupted: false, persistedOutputPath: '/tmp/out.txt', persistedOutputSize: BIG.length },
      text: preview,
    }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n handler src' })

    const bash = result.result as { stdout: string; persistedOutputPath?: string }
    expect(bash.stdout).toBe(SUMMARY)
    expect(bash.persistedOutputPath).toBeUndefined()
    expect(JSON.parse(calls.find(call => call.args[0] === 'process')?.stdin ?? '{}').tool_response.output).toBe(BIG)
    expect(logs.some(line => line.includes('4,200 tokens saved vs full output (5,000 → 800)'))).toBe(true)
  })

  test('output Claude Code saved to a file includes stderr in the engine event', async ($, on) => {
    const { calls } = world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })))
    const preview = '<persisted-output>\nOutput too large (14KB). Full output saved to: /tmp/out.txt\n</persisted-output>'
    on('tool.call', { tool: 'Bash' }, () => ({
      result: { stdout: preview, stderr: 'sandbox: write denied', interrupted: false, persistedOutputPath: '/tmp/out.txt', persistedOutputSize: BIG.length },
      text: preview,
    }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n handler src' })

    expect((result.result as { stdout: string }).stdout).toBe(SUMMARY)
    expect(JSON.parse(calls.find(call => call.args[0] === 'process')?.stdin ?? '{}').tool_response.output)
      .toBe(`${BIG}\nsandbox: write denied`)
  })

  test('the session total adds up across compressed results', async ($, on) => {
    const { logs } = world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })))
    on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: BIG, stderr: '', interrupted: false }, text: BIG }))
    await $.session.start(start)

    await $.tool.call({ tool: 'Bash', command: 'rg -n handler src' })
    await $.tool.call({ tool: 'Bash', command: 'rg -n handler lib' })

    expect(logs.some(line => line.endsWith('session 8,400'))).toBe(true)
  })

  test('small output never reaches the engine', async ($, on) => {
    const { calls } = world(on)
    on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: 'ok\n', stderr: '', interrupted: false }, text: 'ok\n' }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'git status' })

    expect((result.result as { stdout: string }).stdout).toBe('ok\n')
    expect(calls.some(call => call.args[0] === 'process')).toBe(false)
  })

  test('output under the engine minimum in bytes never reaches the engine', async ($, on) => {
    const { calls } = world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })), 5000)
    const text = 'a'.repeat(4900)
    on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: text, stderr: '', interrupted: false }, text }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n a src' })

    expect((result.result as { stdout: string }).stdout).toBe(text)
    expect(calls.some(call => call.args[0] === 'process')).toBe(false)
  })

  test('short but token-dense output reaches the engine', async ($, on) => {
    world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })), 1000)
    const text = '漢字'.repeat(600)
    on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: text, stderr: '', interrupted: false }, text }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n 漢 docs' })

    expect((result.result as { stdout: string }).stdout).toBe(SUMMARY)
  })

  test('context from hooks beneath stays with the summary', async ($, on) => {
    world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })))
    on('tool.call', { tool: 'Bash' }, () =>
      ({ result: { stdout: BIG, stderr: '', interrupted: false }, text: BIG, context: ['note from a hook beneath'] }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n handler src' })

    expect((result.result as { stdout: string }).stdout).toBe(SUMMARY)
    expect(result.context).toEqual(['note from a hook beneath'])
  })

  test('output the engine declines stays exactly as the tool wrote it', async ($, on) => {
    world(on, () => ran(''))
    on('tool.call', { tool: 'Bash' }, () => ({ result: { stdout: BIG, stderr: '', interrupted: false }, text: BIG }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'rg -n handler src' })

    expect((result.result as { stdout: string }).stdout).toBe(BIG)
  })

  test('failed commands are left to the model untouched', async ($, on) => {
    const { calls } = world(on)
    on('tool.call', { tool: 'Bash' }, () => ({ isError: true, text: BIG, result: undefined }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'Bash', command: 'npm test' })

    expect(result.isError).toBe(true)
    expect(calls.some(call => call.args[0] === 'process')).toBe(false)
  })

  test('an MCP text result becomes one summary text block', async ($, on) => {
    const { calls } = world(on, () => ran(JSON.stringify({ continue: false, stopReason: SUMMARY })))
    on('tool.call', { tool: 'mcp__codebase-memory-mcp__search_graph' }, () =>
      ({ result: [{ type: 'text', text: BIG }], text: BIG }))
    await $.session.start(start)

    const result = await $.tool.call({ tool: 'mcp__codebase-memory-mcp__search_graph', name_pattern: 'handler' })

    expect(result.result).toEqual([{ type: 'text', text: SUMMARY }])
    const event = JSON.parse(calls.find(call => call.args[0] === 'process')?.stdin ?? '{}')
    expect(event.tool_response).toEqual({ content: [{ type: 'text', text: BIG }] })
  })
})

describe('options', () => {
  test('the defaults run medium effort with shell summaries on', async ($, on) => {
    const { calls } = world(on)
    await $.session.start(start)

    expect(calls[0]?.env).toEqual({ CONTEXT_SAVER_EFFORT: 'medium', CONTEXT_SAVER_SUMMARIZE_SHELL: '1' })
  })

  test('/config values reach the engine', { options: { effort: 'none', summarize_shell: false } }, async ($, on) => {
    const { calls } = world(on)
    await $.session.start(start)

    expect(calls[0]?.env).toEqual({ CONTEXT_SAVER_EFFORT: 'none', CONTEXT_SAVER_SUMMARIZE_SHELL: '0' })
  })
})

describe('/context-saver', () => {
  test('stats runs the engine report for the asked number of days', async ($, on) => {
    const { calls } = world(on, () => ran('{"compressed_outputs": 3}'))
    await $.session.start(start)

    const answer = await $.command.run(typed('stats 30'))

    expect(calls.at(-1)?.args).toEqual(['stats', '--days', '30'])
    expect(answer.text).toBe('{"compressed_outputs": 3}')
  })

  test('unknown arguments answer with the usage line', async ($, on) => {
    world(on)
    await $.session.start(start)

    const answer = await $.command.run(typed('compress-everything'))

    expect(answer.text).toContain('Usage: /context-saver')
  })
})

describe('savings label', () => {
  test('parses saved, original, producer and cache hits', () => {
    expect(parseSaving('[12 tokens saved by RTK (local)] (estimated context tokens; cache hit)\n' +
      '[Context summary via rtk: logs; exit_code=0; estimated original tokens=40]'))
      .toEqual({ saved: 12, before: 40, producer: 'RTK (local)', isCacheHit: true })
    expect(parseSaving('plain tool output')).toBeUndefined()
  })

  test('counts UTF-8 bytes', () => {
    expect(utf8Bytes('ab')).toBe(2)
    expect(utf8Bytes('é漢')).toBe(5)
    expect(utf8Bytes('😀')).toBe(4)
  })

  test('maps command arguments to the engine CLI', () => {
    expect(commandArgv('')).toEqual(['stats', '--days', '7'])
    expect(commandArgv('review looked fine')).toEqual(['review', '--note', 'looked fine'])
    expect(commandArgv('stats seven')).toBeUndefined()
    expect(commandArgv('disable now')).toBeUndefined()
  })
})
