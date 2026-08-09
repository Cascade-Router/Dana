# Dana UI Assets

Brand icons are centralized at the project root:

- ``assets/dana_logo.png``
- ``assets/dana_logo.ico``

Regenerate via ``python scripts/generate_dana_icon.py``.

This directory may hold optional UI-only drop-ins; runtime logo resolution
prefers the root ``assets/`` SSoT through ``dana.resources.get_resource_path``.
