/**
 * Las partes de cada capítulo, en el orden en que se leen.
 *
 * Base.astro monta con esto la barra de navegación que aparece arriba de todas
 * las páginas del capítulo, así que un capítulo nuevo solo necesita añadirse
 * aquí. Los capítulos de una sola parte no la muestran: no habría a dónde ir.
 */
export interface Parte {
  href: string;
  nombre: string;
  /** Aclara de qué va, cuando el nombre solo no basta. */
  pie?: string;
}

export const CAPITULOS: Record<string, Parte[]> = {
  ch01: [
    { href: "/ch01", nombre: "El panorama", pie: "conceptos" },
    { href: "/ch01/lifesat", nombre: "lifesat", pie: "dataset" },
    { href: "/ch01/lifesat/demo", nombre: "Recta contra vecinos", pie: "demo" },
  ],
  ch02: [
    { href: "/ch02", nombre: "Punta a punta", pie: "método" },
    { href: "/ch02/california-housing", nombre: "California Housing", pie: "dataset" },
    { href: "/ch02/california-housing/demo", nombre: "Dónde se equivoca", pie: "demo" },
  ],
  ch03: [{ href: "/ch03", nombre: "MNIST", pie: "clasificación" }],
};

/** Sin barra dentro de la barra: una sola parte no es navegación. */
export const tieneNavegacion = (cap: string) => (CAPITULOS[cap]?.length ?? 0) > 1;
