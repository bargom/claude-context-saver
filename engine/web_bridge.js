// Load this trusted local file with new Function('tools', source)(tools) in
// functions.exec; only emit the returned web result, never the captured source.
// Native web is not intercepted globally by Codex hooks.
// Print it with `context-saver web-bridge`: that fills in these paths.
// The launcher runs the engine in its venv with its tiktoken cache.
const LAUNCHER = '__CONTEXT_SAVER_LAUNCHER__';
const STATE = '__CONTEXT_SAVER_STATE__';
return async function summarizedWebRun(request, options = {}) {
  const original = await tools.web__run(request);
  if (options.raw === true) return original;
  let stagedPath;
  const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
  const complete = async args => {
    let response = await tools.exec_command(args);
    let output = response.output || '';
    while (response.session_id !== undefined) {
      response = await tools.write_stdin({session_id: response.session_id, chars: '',
        max_output_tokens: args.max_output_tokens, yield_time_ms: 10000});
      output += response.output || '';
    }
    return {...response, output};
  };
  try {
    const deliveryToken = 'web_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2);
    const envelope = JSON.stringify({
      tool_name: 'web.run',
      tool_input: request,
      tool_response: original,
      session_id: options.sessionId || '',
      tool_use_id: options.toolUseId || '',
      delivery_token: deliveryToken,
    });
    // Avoid large process arguments and rich payloads; the Python adapter also
    // enforces its source size limit and preserves unsupported content.
    if (envelope.length > 4 * 1024 * 1024) return original;
    const adapter = quote(LAUNCHER) + ' web-adapter --decision-only';
    let command;
    if (envelope.length <= 24000) {
      command = "printf '%s' " + quote(envelope) + ' | ' + adapter;
    } else {
      // Short, quoted chunks avoid ARG_MAX; originals never enter model context.
      const created = await complete({
        cmd: 'umask 077\nmktemp ' + quote(STATE + '/web-input.XXXXXXXX'),
        max_output_tokens: 100, yield_time_ms: 1000,
      });
      const candidate = (created.output || '').trim();
      if (created.exit_code !== 0 || created.session_id !== undefined ||
          !candidate.startsWith(STATE + '/web-input.') ||
          !/^[A-Za-z0-9]+$/.test(candidate.slice((STATE + '/web-input.').length))) return original;
      stagedPath = candidate;
      for (let offset = 0; offset < envelope.length;) {
        let end = Math.min(offset + 24000, envelope.length);
        // Keep UTF-16 surrogate pairs intact across independently encoded writes.
        if (end < envelope.length && /[\uD800-\uDBFF]/.test(envelope[end - 1])) end--;
        const written = await complete({
          cmd: "printf '%s' " + quote(envelope.slice(offset, end)) + ' >> ' + quote(stagedPath),
          max_output_tokens: 100, yield_time_ms: 1000,
        });
        if (written.exit_code !== 0 || written.session_id !== undefined) return original;
        offset = end;
      }
      command = adapter + ' < ' + quote(stagedPath);
    }
    const response = await complete({cmd: command, max_output_tokens: 5000, yield_time_ms: 1000});
    if (response.exit_code !== 0) return original;
    const decision = JSON.parse(response.output);
    if (decision.compressed === true && typeof decision.result === 'string' && decision.result.trim()) {
      // Count savings only after a complete adapter response reached this
      // bridge; a lost acknowledgment conservatively undercounts delivery.
      try {
        await complete({
          cmd: quote(LAUNCHER) + ' delivered --delivery-token ' + quote(deliveryToken),
          max_output_tokens: 100, yield_time_ms: 1000,
        });
      } catch (_) { /* Delivery still succeeds; its savings remain unconfirmed. */ }
      return decision.result;
    }
  } catch (_) {
    // Keep the original in this isolate so truncation, helper errors, and
    // transport failures cannot replace it with a partial fallback.
  } finally {
    if (stagedPath) {
      try {
        await complete({cmd: 'rm -f -- ' + quote(stagedPath), max_output_tokens: 100, yield_time_ms: 1000});
      } catch (_) { /* A transport failure may leave this private staging file. */ }
    }
  }
  return original;
};
