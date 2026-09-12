from unidecompiler.plugins import FrontendVersionSupport

VERSION_SUPPORT = FrontendVersionSupport(
    family="qimeivm",
    versions=('1',),
    parser="qimeivm-json-int-stream-v1",
    status="experimental",
)
