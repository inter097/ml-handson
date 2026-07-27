// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  // Sitio estático: ninguna página necesita backend. Las predicciones del
  // capítulo 2 se calculan una vez con Python y viajan como JSON.
  output: "static",
  site: "https://ml.eliuth.dev",
  build: { format: "directory" },
});
