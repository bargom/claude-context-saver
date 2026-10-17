/** One compressed result, read from the label the engine puts on its first line. */
export type Saving = { saved: number; before: number; producer: string; isCacheHit: boolean }

// Written by engine/hook.py label_savings() and render_replacement().
const LABEL = /^\[(\d+) tokens saved by (.+?)\] \(estimated context tokens(; cache hit)?\)/
const ORIGINAL = /estimated original tokens=(\d+)/

export function parseSaving(replacement: string): Saving | undefined {
  const label = LABEL.exec(replacement)
  if (!label || label[1] === undefined || label[2] === undefined) return undefined
  const saved = Number(label[1])
  const original = ORIGINAL.exec(replacement)?.[1]
  return { saved, before: original === undefined ? saved : Number(original), producer: label[2], isCacheHit: label[3] !== undefined }
}

/** UTF-8 length. A byte-level BPE token covers at least one byte, so tokens <= bytes. */
export function utf8Bytes(text: string): number {
  let bytes = 0
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i)
    if (code < 0x80) bytes += 1
    else if (code < 0x800) bytes += 2
    else if (code >= 0xd800 && code < 0xdc00 && i + 1 < text.length) { bytes += 4; i++ }
    else bytes += 3
  }
  return bytes
}

/** 12345 -> "12,345" without relying on Intl in the hooks environment. */
export const formatCount = (value: number): string => String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',')

/**
 * The dim transcript line shown for every compressed result. isVsFullOutput:
 * Claude Code had saved the output to a file and would have shown only a
 * preview, so the count is against the full output.
 */
export function savingLine(saving: Saving, sessionTokens: number, isVsFullOutput = false): string {
  const after = Math.max(0, saving.before - saving.saved)
  return `⛁ context-saver: ${formatCount(saving.saved)} tokens saved${isVsFullOutput ? ' vs full output' : ''} ` +
    `(${formatCount(saving.before)} → ${formatCount(after)}) by ${saving.producer}` +
    `${saving.isCacheHit ? ', cache hit' : ''} · session ${formatCount(sessionTokens)}`
}

/** `/context-saver <args>` to the engine CLI's argv; undefined for unknown input. */
export function commandArgv(args: string): string[] | undefined {
  const [action = 'stats', ...rest] = args.trim().split(/\s+/).filter(Boolean)
  // At least one day: the engine reads an omitted --days as all time.
  if (action === 'stats') return /^[1-9]\d{0,3}$/.test(rest[0] ?? '7') ? ['stats', '--days', rest[0] ?? '7'] : undefined
  if (['status', 'setup', 'enable', 'disable'].includes(action) && rest.length === 0) return [action]
  // One argument, so a note that starts with "-" is not read as an option.
  if (action === 'review' && rest.length > 0) return ['review', `--note=${rest.join(' ')}`]
  return undefined
}

export const COMMAND_USAGE = 'Usage: /context-saver [stats [days] | status | setup | enable | disable | review <note>]'
