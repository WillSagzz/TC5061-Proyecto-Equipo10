---
titulo: "TC5061 · Proyecto MiBici: Microservicios, CI/CD y FinOps · Equipo 10"
archivo_salida: TC5061_Operacion_FinOps_Equipo10
---

<!-- Compilar con `make entrega06`. Reporte de 3 a 5 cuartillas (secciones 01 a 04) más la portada
     y los anexos. Una sección por archivo. -->

<style>
/* Formato que pide esta entrega: Arial 11 e interlineado 1.15. Solo aplica a este documento;
   docs/entregas/estilos.css y los PDF de las entregas anteriores no cambian. */
body { font-family: Arial, Helvetica, sans-serif; line-height: 1.15; }
h1 { font-size: 15pt; margin: 0.9em 0 0.4em; }
h2 { font-size: 12.5pt; margin: 0.9em 0 0.35em; }
h1#anexos::before { content: none; }  /* los anexos no llevan número de sección */
figure img { max-height: 6.5cm; }
figure img[src*="locust_01"] { max-height: 9cm; }  /* tres paneles: necesita más altura */
figure img[src*="actions_01"] { max-height: 5.5cm; }
figure { margin: 0.5em 0 0.8em; }
table { break-inside: avoid; }  /* ninguna tabla se parte entre dos páginas */
</style>

{!secciones/00-portada.md!}

<!-- pagebreak -->

{!secciones/01-orquestacion.md!}

{!secciones/02-pipeline.md!}

{!secciones/03-pruebas-carga.md!}

{!secciones/04-finops.md!}

<!-- pagebreak -->

{!secciones/05-anexos.md!}
