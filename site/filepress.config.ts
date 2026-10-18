import { defineFilepressConfig } from 'getfilepress';

const github = 'https://github.com/Catalyst-Forge-LLC/jjm';

export default defineFilepressConfig({
	title: 'JJM',
	description: "Give JJM the current state and the options your application allows. It returns weights, provenance, and pass, hold, or reject against your declared requirements. Your application decides what happens next.",
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
