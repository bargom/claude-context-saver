/** Tokens saved in this session: compressed outputs and their summed savings. */
export type SavingsTotal = { outputs: number; tokens: number }

declare module 'claude-code' {
  interface PluginState {
    'context-saver': { session: SavingsTotal }
  }
}
