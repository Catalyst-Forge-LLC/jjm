import { defineFilepressConfig } from 'getfilepress';

const github = 'https://github.com/Catalyst-Forge-LLC/jjm';

export default defineFilepressConfig({
	title: 'JJM',
	description: "Use JJM when an application must choose among fixed options, such as assigning support messages to teams. Fit a local decision model on labeled examples, then receive a selected option, weights, provenance, and acceptance checks your application can use to route or request review.",
	tagline: 'Just judgment.',
	lede: "Local judgment runtime for developers · alpha",
	url: 'https://jjm.dev',
	author: 'Catalyst Forge LLC',
	logo: '/wordmark.svg',
	ogImage: '/og.png',
	homePage: 'about',
	nav: [
		{ label: 'Home', href: '/' },
		{ label: 'Benchmark', href: '/routing-benchmark' },
		{ label: 'CPU backend', href: '/tfidf' },
		{ label: 'GitHub', href: github, icon: 'github' }
	],
	footerLinks: [
		{ label: 'See the rest of the Catalyst Forge shelf.', href: 'https://catalystforge.com/tools/' },
		{ label: 'GitHub', href: github, icon: 'github' }
	]
});
