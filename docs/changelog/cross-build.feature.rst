Added experimental support for cross builds. The new ``--build-details`` option takes a :pep:`739`
``build-details.json`` describing the target Python; ``build`` then installs the new
``[build-system] host-requires`` (plus the backend's ``get_host_requires_for_cross_build_wheel()``)
for the target platform into a directory and invokes the backend's ``cross_build_wheel()`` hook.
``--toolchain`` and ``--host-glibc-version`` describe the cross toolchain and the target's glibc.
