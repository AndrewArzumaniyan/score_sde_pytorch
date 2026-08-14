# Third-party components

## NVLabs EDM

The canonical EDM configs load the official implementation from:

- Repository: https://github.com/NVlabs/edm
- Commit: `008a4e5316c8e3bfe61a62f874bddba254295afb`
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0

The source is installed into the ignored `third_party/edm/` directory by
`tools/install_official_edm.sh`; it is not copied into or relicensed as part of
this Apache-licensed repository. The adapter is implemented in
`models/edm_canonical.py`.

The Docker build runs the same installer and therefore includes that pinned
checkout in the resulting image under its original license. A runtime bind
mount of the repository hides the image copy at that path, so the host checkout
must also contain the pinned `third_party/edm/` directory.

The official license restricts use to non-commercial purposes and imposes
attribution/share-alike conditions. Consult the installed `LICENSE.txt` for
the complete terms.
