
import os
import bayesik as bik


# Export all notebooks to HTML_EMBED format:
dirIPYNB    = os.path.join( bik.dirREPO, 'Appendix', 'ipynb' )
dirHTML     = os.path.join( bik.dirREPO, 'Appendix', 'html' )
for root,dir0,fnames in os.walk(dirIPYNB):
	for name in fnames:
		if name.endswith('.ipynb'):
			print(f'Exporting {name}...')
			fnameIPYNB = os.path.join(dirIPYNB, name)
			fnameHTML  = os.path.join(dirHTML, name[:-6] + '.html')
			cmd        = f'jupyter nbconvert --to html_embed {fnameIPYNB} --output {fnameHTML}'
			os.system( cmd )
			print('\n\n\n')
print('Done.')






