import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const from = 'name="twitter:card" content="summary"';
const to = 'name="twitter:card" content="summary_large_image"';

async function walk(dir) {
	for (const entry of await readdir(dir, { withFileTypes: true })) {
		const path = join(dir, entry.name);
		if (entry.isDirectory()) {
			await walk(path);
			continue;
		}
		if (!entry.name.endsWith('.html')) continue;
		const text = await readFile(path, 'utf8');
		if (!text.includes(from)) continue;
		await writeFile(path, text.replaceAll(from, to));
	}
}

await walk('build');
