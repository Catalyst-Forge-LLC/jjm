# jjm.dev

Public alpha page for [Just Judgment Machine](https://github.com/Catalyst-Forge-LLC/jjm). [FilePress](https://getfilepress.com) (`getfilepress` on npm).

The page states the same limits as the repository README. It does not host inference, and it does not offer a package install.

```bash
pnpm install
pnpm dev
pnpm build
```

## Deploy (Cloudflare Pages)

```bash
pnpm ship
```

That builds this folder and uploads it to the Pages project `jjm`. Attach **jjm.dev** to that project. One pipeline only.
