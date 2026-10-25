// The cartesian dist bundle has the same API as plotly.js, restricted to cartesian traces
// (scatter, bar, heatmap, histogram, box): reuse the plotly.js types for it.
declare module "plotly.js-cartesian-dist-min" {
  import * as Plotly from "plotly.js";
  export default Plotly;
}
