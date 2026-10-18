"""Compila una entrega escrita en varios archivos Markdown a un solo .md, .html y .pdf.

Uso (desde la raiz del repositorio):

    python docs/entregas/build.py docs/entregas/entrega01            # md + html + pdf
    python docs/entregas/build.py docs/entregas/entrega01 --sin-pdf  # solo md + html
    python docs/entregas/build.py docs/entregas/entrega01 --abrir    # abre el pdf al terminar

Cada entrega tiene un archivo maestro ``entregable.md`` con metadatos (front matter YAML) y
directivas de inclusion con la sintaxis ``{!ruta/relativa.md!}``. El script:

1. Resuelve las inclusiones (recursivas, relativas al archivo que incluye).
2. Reescribe las rutas de imagenes para que sigan funcionando desde ``build/``.
3. Reemplaza los marcadores ``<!-- pagebreak -->`` y ``{{toc}}``.
4. Genera ``build/<archivo_salida>.md`` (Markdown compilado, util para revisar en GitHub).
5. Renderiza HTML con markdown-it-py y la hoja ``docs/entregas/estilos.css``.
6. Imprime el HTML a PDF con un navegador Chromium headless (Edge o Chrome), o con pandoc
   si no encuentra navegador.

Dependencias de Python: markdown-it-py y PyYAML (ambas fijadas en requirements.txt).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import webbrowser

from markdown_it import MarkdownIt
import yaml

AQUI = Path(__file__).resolve().parent
CSS_POR_DEFECTO = AQUI / "estilos.css"

RE_INCLUDE = re.compile(r"^\{!\s*(?P<ruta>[^!]+?)\s*!\}\s*$", re.MULTILINE)
RE_FRONT_MATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n", re.DOTALL)
RE_IMG_MD = re.compile(r"(!\[[^\]]*\]\()(?P<ruta>[^)\s]+)(\s*(?:\"[^\"]*\")?\))")
RE_IMG_HTML = re.compile(r"(<img\b[^>]*?\bsrc=[\"'])(?P<ruta>[^\"']+)([\"'])")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)
RE_FENCE = re.compile(r"^(```|~~~)")
RE_COMENTARIO = re.compile(r"(<!--.*?-->)", re.DOTALL)

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"


# ----------------------------------------------------------------------------- utilidades


def leer(ruta: Path) -> str:
    return ruta.read_text(encoding="utf-8")


def es_ruta_externa(ruta: str) -> bool:
    return bool(re.match(r"^(?:[a-z]+:)?//|^data:|^mailto:|^#|^/", ruta, re.IGNORECASE))


def slug(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    texto = re.sub(r"[^\w\s-]", "", texto).strip().lower()
    return re.sub(r"[\s_]+", "-", texto)


def reescribir_imagenes(markdown: str, base_origen: Path, base_destino: Path) -> str:
    """Convierte rutas de imagen relativas a *base_origen* en rutas relativas a *base_destino*."""

    def _reescribir(m: re.Match) -> str:
        ruta = m.group("ruta")
        if es_ruta_externa(ruta):
            return m.group(0)
        absoluta = (base_origen / ruta).resolve()
        if not absoluta.exists():
            print(f"  [aviso] imagen no encontrada: {absoluta}", file=sys.stderr)
        relativa = os.path.relpath(absoluta, base_destino).replace(os.sep, "/")
        return f"{m.group(1)}{relativa}{m.group(3)}"

    # Los comentarios HTML (por ejemplo, figuras aun no disponibles) se dejan intactos.
    partes = RE_COMENTARIO.split(markdown)
    for i in range(0, len(partes), 2):
        partes[i] = RE_IMG_HTML.sub(_reescribir, RE_IMG_MD.sub(_reescribir, partes[i]))
    return "".join(partes)


# ----------------------------------------------------------------------------- inclusiones


def quitar_front_matter(texto: str) -> tuple[dict, str]:
    m = RE_FRONT_MATTER.match(texto)
    if not m:
        return {}, texto
    meta = yaml.safe_load(m.group("yaml")) or {}
    return meta, texto[m.end() :]


def resolver_includes(texto: str, archivo: Path, base_destino: Path, pila: tuple = ()) -> str:
    """Sustituye cada ``{!ruta!}`` por el contenido del archivo, de forma recursiva."""
    if archivo in pila:
        cadena = " -> ".join(str(p) for p in pila + (archivo,))
        raise RuntimeError(f"Inclusion circular detectada: {cadena}")

    def _incluir(m: re.Match) -> str:
        ruta = (archivo.parent / m.group("ruta").strip()).resolve()
        if not ruta.exists():
            raise FileNotFoundError(f"{archivo.name}: no existe el archivo incluido {ruta}")
        print(f"  + {m.group('ruta').strip()}")
        contenido = quitar_front_matter(leer(ruta))[1]
        contenido = resolver_includes(contenido, ruta, base_destino, pila + (archivo,))
        contenido = reescribir_imagenes(contenido, ruta.parent, base_destino)
        return contenido.strip("\n") + "\n"

    return RE_INCLUDE.sub(_incluir, texto)


# ----------------------------------------------------------------------------- marcadores


def generar_toc(markdown: str, nivel_max: int = 2) -> str:
    """Tabla de contenido en Markdown a partir de los encabezados (ignora bloques de codigo)."""
    lineas = []
    dentro_fence = False
    for linea in markdown.splitlines():
        if RE_FENCE.match(linea):
            dentro_fence = not dentro_fence
            continue
        if dentro_fence:
            continue
        m = RE_HEADING.match(linea)
        if not m:
            continue
        nivel = len(m.group(1))
        if nivel > nivel_max or "{.sin-toc}" in m.group(2):
            continue
        titulo = re.sub(r"\s*\{[^}]*\}\s*$", "", m.group(2)).strip()
        sangria = "  " * (nivel - 1)
        lineas.append(f"{sangria}- [{titulo}](#{slug(titulo)})")
    return "\n".join(lineas) + "\n"


def aplicar_marcadores(markdown: str) -> str:
    markdown = markdown.replace("<!-- pagebreak -->", '<div class="pagebreak"></div>')
    if "{{toc}}" in markdown:
        toc = generar_toc(markdown.replace("{{toc}}", ""))
        markdown = markdown.replace("{{toc}}", '<nav class="toc">\n\n' + toc + "\n</nav>")
    # quita atributos de encabezado tipo {.sin-toc} antes de renderizar
    markdown = re.sub(r"^(#{1,6}\s+.+?)\s*\{[^}]*\}\s*$", r"\1", markdown, flags=re.MULTILINE)
    return markdown


# ----------------------------------------------------------------------------- render HTML


def crear_renderer() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True, "typographer": False})
    md.enable(["table", "strikethrough"])
    renderer = md.renderer
    fence_original = renderer.rules.get("fence")

    def heading_open(tokens, idx, options, env):
        token = tokens[idx]
        inline = tokens[idx + 1]
        hijos = inline.children or []
        texto = "".join(t.content for t in hijos if t.type in ("text", "code_inline"))
        token.attrSet("id", slug(texto))
        return renderer.renderToken(tokens, idx, options, env)

    def fence(tokens, idx, options, env):
        token = tokens[idx]
        lenguaje = (token.info or "").strip().split(" ")[0]
        if lenguaje == "mermaid":
            env["usa_mermaid"] = True
            return f'<pre class="mermaid">\n{token.content}</pre>\n'
        if fence_original is not None:
            return fence_original(tokens, idx, options, env)
        return renderer.fence(tokens, idx, options, env)

    def image(tokens, idx, options, env):
        token = tokens[idx]
        alt = renderer.renderInlineAsText(token.children or [], options, env)
        token.attrSet("alt", alt)
        html = f"<img{renderer.renderAttrs(token)}>"
        if token.meta.get("figura") and alt:
            html += f"<figcaption>{alt}</figcaption>"
        return html

    def marcar_figuras(state):
        toks = state.tokens
        for i in range(len(toks) - 2):
            hijos = toks[i + 1].children or []
            if (
                toks[i].type == "paragraph_open"
                and toks[i + 1].type == "inline"
                and toks[i + 2].type == "paragraph_close"
                and len(hijos) == 1
                and hijos[0].type == "image"
            ):
                toks[i].tag = toks[i + 2].tag = "figure"
                hijos[0].meta["figura"] = True

    renderer.rules["heading_open"] = heading_open
    renderer.rules["fence"] = fence
    renderer.rules["image"] = image
    md.core.ruler.push("marcar_figuras", marcar_figuras)
    return md


PLANTILLA_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{titulo}</title>
<style>
{css}
</style>
</head>
<body>
<main class="documento">
{cuerpo}
</main>
{scripts}
</body>
</html>
"""

SCRIPT_MERMAID = (
    f'<script src="{MERMAID_CDN}"></script>\n'
    "<script>mermaid.initialize({ startOnLoad: true, theme: \"neutral\", securityLevel: \"loose\","
    ' themeVariables: { fontSize: "22px", fontFamily: "Segoe UI, Arial, sans-serif" },'
    " flowchart: { useMaxWidth: true, htmlLabels: true, nodeSpacing: 35, rankSpacing: 45, padding: 12 } });"
    "</script>"
)


def renderizar_html(markdown: str, titulo: str, css: str) -> str:
    md = crear_renderer()
    env: dict = {}
    cuerpo = md.render(markdown, env)
    scripts = SCRIPT_MERMAID if env.get("usa_mermaid") else ""
    return PLANTILLA_HTML.format(titulo=titulo, css=css, cuerpo=cuerpo, scripts=scripts)


# ----------------------------------------------------------------------------- PDF


def buscar_navegador() -> str | None:
    """Devuelve la ruta a un navegador Chromium (Edge o Chrome) o None."""
    for variable in ("NAVEGADOR_PDF", "CHROME_PATH", "BROWSER"):
        valor = os.environ.get(variable)
        if valor and Path(valor).exists():
            return valor
    candidatos = [
        # Windows
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        str(Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe"),
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ]
    for c in candidatos:
        if Path(c).exists():
            return c
    nombres = ("google-chrome", "chrome", "chromium", "chromium-browser", "microsoft-edge", "msedge")
    for nombre in nombres:
        ruta = shutil.which(nombre)
        if ruta:
            return ruta
    return None


def esperar_pdf(pdf: Path, segundos: float = 120.0) -> bool:
    """Espera a que el PDF exista y deje de crecer.

    En Windows, msedge.exe devuelve el control de inmediato y escribe el archivo en un proceso
    hijo, asi que no basta con esperar a que termine el comando.
    """
    limite = time.monotonic() + segundos
    tamano_previo = -1
    estable = 0
    while time.monotonic() < limite:
        if pdf.exists():
            tamano = pdf.stat().st_size
            if tamano > 0 and tamano == tamano_previo:
                estable += 1
                if estable >= 3:  # ~1.5 s sin cambios
                    return True
            else:
                estable = 0
            tamano_previo = tamano
        time.sleep(0.5)
    return pdf.exists() and pdf.stat().st_size > 0


def html_a_pdf(html: Path, pdf: Path) -> bool:
    navegador = buscar_navegador()
    if navegador:
        # Perfil propio y persistente (dentro de build/, ignorado por Git) para no chocar con una
        # ventana del navegador que el usuario tenga abierta.
        perfil = pdf.parent / ".perfil-navegador"
        perfil.mkdir(exist_ok=True)
        pdf.unlink(missing_ok=True)
        comando = [
            navegador,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={perfil}",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf}",
            html.resolve().as_uri(),
        ]
        print(f"  navegador: {navegador}")
        resultado = subprocess.run(comando, capture_output=True, text=True, timeout=180)
        if esperar_pdf(pdf):
            return True
        print(resultado.stdout, resultado.stderr, file=sys.stderr)
    pandoc = shutil.which("pandoc")
    if pandoc:
        print("  navegador no disponible, usando pandoc")
        resultado = subprocess.run([pandoc, str(html), "-o", str(pdf)], capture_output=True, text=True)
        if resultado.returncode == 0:
            return True
        print(resultado.stderr, file=sys.stderr)
    print(
        "  [error] No se pudo generar el PDF. Instala Google Chrome o Microsoft Edge, o define la\n"
        "          variable de entorno NAVEGADOR_PDF con la ruta al ejecutable del navegador.",
        file=sys.stderr,
    )
    return False


# ----------------------------------------------------------------------------- principal


def compilar(entrega: Path, generar_pdf: bool = True, abrir: bool = False) -> int:
    entrega = entrega.resolve()
    maestro = entrega / "entregable.md"
    if not maestro.exists():
        print(f"[error] No existe {maestro}", file=sys.stderr)
        return 1

    salida = entrega / "build"
    salida.mkdir(exist_ok=True)

    def mostrar(ruta: Path) -> str:
        try:
            return str(ruta.relative_to(Path.cwd())).replace(os.sep, "/")
        except ValueError:
            return str(ruta)

    print(f"Compilando {entrega.name}")
    meta, cuerpo = quitar_front_matter(leer(maestro))
    titulo = str(meta.get("titulo", entrega.name))
    nombre = str(meta.get("archivo_salida", entrega.name))
    css = leer(Path(meta["css"]) if meta.get("css") else CSS_POR_DEFECTO)

    # Primero las imagenes del maestro (relativas a la carpeta de la entrega) y despues las
    # inclusiones, que reescriben sus propias imagenes relativas a cada seccion.
    cuerpo = reescribir_imagenes(cuerpo, entrega, salida)
    cuerpo = resolver_includes(cuerpo, maestro, salida)
    cuerpo = aplicar_marcadores(cuerpo)

    md_final = salida / f"{nombre}.md"
    md_final.write_text(cuerpo, encoding="utf-8")
    print(f"  -> {mostrar(md_final)}")

    html_final = salida / f"{nombre}.html"
    html_final.write_text(renderizar_html(cuerpo, titulo, css), encoding="utf-8")
    print(f"  -> {mostrar(html_final)}")

    if not generar_pdf:
        return 0

    pdf_final = salida / f"{nombre}.pdf"
    if not html_a_pdf(html_final, pdf_final):
        return 2
    print(f"  -> {mostrar(pdf_final)}")
    # Copia del PDF final en la raiz de la entrega: build/ esta ignorado por Git y este si se versiona.
    pdf_entrega = entrega / pdf_final.name
    shutil.copy2(pdf_final, pdf_entrega)
    print(f"  -> {mostrar(pdf_entrega)}")
    if abrir:
        webbrowser.open(pdf_final.resolve().as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("entrega", type=Path, help="carpeta de la entrega (contiene entregable.md)")
    parser.add_argument("--sin-pdf", action="store_true", help="solo genera el .md compilado y el .html")
    parser.add_argument("--abrir", action="store_true", help="abre el PDF al terminar")
    args = parser.parse_args(argv)
    return compilar(args.entrega, generar_pdf=not args.sin_pdf, abrir=args.abrir)


if __name__ == "__main__":
    sys.exit(main())
