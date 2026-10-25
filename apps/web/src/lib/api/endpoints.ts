/** One function per endpoint, so components never deal with paths or response envelopes. */
import { client, unwrap, type Schemas } from "./client";

type Opt = { signal?: AbortSignal };

export const api = {
  status: (o: Opt = {}) => unwrap(client.GET("/api/v1/status", { signal: o.signal })),
  universe: (o: Opt = {}) => unwrap(client.GET("/api/v1/universe", { signal: o.signal })),
  demoPortfolios: (o: Opt = {}) =>
    unwrap(client.GET("/api/v1/demo/portfolios", { signal: o.signal })),

  var: (body: Schemas["VaRRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/risk/var", { body, signal: o.signal })),
  metrics: (body: Schemas["MetricsRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/risk/metrics", { body, signal: o.signal })),
  attribution: (body: Schemas["AttributionRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/risk/attribution", { body, signal: o.signal })),
  correlation: (body: Schemas["CorrelationRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/risk/correlation", { body, signal: o.signal })),
  backtest: (body: Schemas["BacktestRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/backtest", { body, signal: o.signal })),
  factors: (body: Schemas["FactorRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/factors/exposures", { body, signal: o.signal })),

  scenarios: (o: Opt = {}) => unwrap(client.GET("/api/v1/stress/scenarios", { signal: o.signal })),
  stress: (body: Schemas["StressRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/stress/run", { body, signal: o.signal })),

  optimize: (body: Schemas["OptimizeRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/optimize", { body, signal: o.signal })),
  goal: (body: Schemas["GoalRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/goals/simulate", { body, signal: o.signal })),
  questionnaire: (o: Opt = {}) =>
    unwrap(client.GET("/api/v1/goals/questionnaire", { signal: o.signal })),
  assistantStatus: (o: Opt = {}) => unwrap(client.GET("/api/v1/ai/status", { signal: o.signal })),
  parseScenario: (body: Schemas["ScenarioParseRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/ai/parse-scenario", { body, signal: o.signal })),
  profile: (body: Schemas["ProfileRequest"], o: Opt = {}) =>
    unwrap(client.POST("/api/v1/goals/profile", { body, signal: o.signal })),
};
