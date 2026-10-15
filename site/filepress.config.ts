import { defineFilepressConfig } from 'getfilepress';

const github = 'https://github.com/Catalyst-Forge-LLC/jjm';

export default defineFilepressConfig({
	title: 'JJM',
	description:
		'Just Judgment Machine is an open-source local judgment runtime. Alpha. Your application makes the decision.',
	tagline: 'Just judgment.',
	lede: 'Alpha · local runtime · explicit criteria',
	url: 'https://jjm.dev',
	author: 'Catalyst Forge LLC',
	logo: '/wordmark.svg',
	ogImage: '/og.png',
	homePage: 'about',
	nav: [
		{ label: 'Home', href: '/' },
		{ label: 'GitHub', href: github, icon: 'github' }
	],
	footerLinks: [
		{ label: 'See the rest of the Catalyst Forge shelf.', href: 'https://catalystforge.com/tools/' },
		{ label: 'GitHub', href: github, icon: 'github' }
	]
});
