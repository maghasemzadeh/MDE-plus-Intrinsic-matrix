from pathlib import Path
import re, json, shutil
from weasyprint import HTML
src = Path('deliverables/thesis-defense-slides/camera-intrinsics-thesis-defense.html')
out_html = Path('/private/tmp/camera-intrinsics-thesis-defense-print.html')
out_pdf = Path('/private/tmp/camera-intrinsics-thesis-defense-full.pdf')
s = src.read_text()
m = re.search(r'window\.__DECK_IMAGES__\s*=\s*(\{.*?\});', s, re.S)
if not m: raise SystemExit('embedded image dictionary not found')
images = json.loads(m.group(1))
for img_id,key in {'img-scale-foc':'scale_foc','img-kswp-abl':'kswp_abl','img-kswp-r2':'kswp_r2','img-kswp-foc':'kswp_foc','img-q-input':'qual_input','img-q-gt':'qual_gt','img-q-base':'qual_base','img-q-prop':'qual_prop'}.items():
 pat=re.compile(r'(<img\s+id="'+re.escape(img_id)+r'")([^>]*>)')
 s,n=pat.subn(lambda mm:mm.group(1)+' src="'+images[key]+'"'+mm.group(2),s,count=1)
 if n != 1: raise SystemExit(f'image tag {img_id} not replaced')
poster=(src.parent/'assets/little-big-ambiguty-poster.jpg').resolve().as_uri()
s=s.replace('src="assets/little-big-ambiguty-poster.jpg"',f'src="{poster}"')
s=re.sub(r'<script\b[^>]*>.*?</script>','',s,flags=re.S|re.I)
s=re.sub(r'<link[^>]+fonts\.googleapis\.com[^>]*>','',s,flags=re.I)
css='''@page { size: 13.333333in 7.5in; margin: 0; }
html, body { width:1280px; height:auto; margin:0; padding:0; overflow:visible !important; background:#0a0c11; }
.deck-viewport { position:static !important; inset:auto !important; overflow:visible !important; width:1280px !important; height:auto !important; }
.deck-track { display:block !important; position:static !important; width:1280px !important; height:auto !important; transform:none !important; transition:none !important; }
.slide { display:block !important; position:relative !important; flex:none !important; width:1280px !important; height:720px !important; min-height:720px !important; max-height:720px !important; overflow:hidden !important; break-after:page; page-break-after:always; }
.slide:last-child { break-after:auto; page-break-after:auto; }
.slide-inner { position:absolute !important; inset:0 !important; width:1280px !important; height:720px !important; }
#progress,#hud,#hint,#overview,#notes,#video-modal { display:none !important; }
'''
s=s.replace('</style>',css+'</style>',1)
out_html.write_text(s)
doc=HTML(filename=str(out_html),base_url=str(src.parent.resolve())).render()
print('pages',len(doc.pages))
doc.write_pdf(str(out_pdf))
shutil.copy2(out_pdf,Path('deliverables/thesis-defense-slides/camera-intrinsics-thesis-defense.pdf'))
print('bytes',Path('deliverables/thesis-defense-slides/camera-intrinsics-thesis-defense.pdf').stat().st_size)
