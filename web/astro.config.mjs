// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  // Sitio estático: ninguna página necesita backend. Las predicciones del
  // capítulo 2 se calculan una vez con Python y viajan como JSON.
  output: "static",
  site: "https://ml.eliuth.dev",
  build: { format: "directory" },

  // Astro lo trae activado, pero colapsa el salto de línea que precede a una
  // etiqueta en línea y pega la palabra con la siguiente: «de ahí la<strong>
  // validación cruzada» salía como «lavalidación». Lo que ahorra es ruido
  // frente a gzip; lo que rompe es el texto.
  compressHTML: false,
});
