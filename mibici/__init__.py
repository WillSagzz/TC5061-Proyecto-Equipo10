import warnings

# frictionless exige chardet >= 6 y requests 2.32 avisa de esa versión al importarse; el aviso
# es cosmético (requests funciona igual) y se silencia antes de que cualquier módulo lo importe.
warnings.filterwarnings(
    "ignore", message=r"urllib3 .* or chardet .* doesn't match a supported version"
)
