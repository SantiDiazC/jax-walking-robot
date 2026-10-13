"""Audit tracked source size, generated files, notebook outputs and local links."""
import json
from pathlib import Path
import re
import subprocess


GENERATED={'.png','.gif','.mp4','.npz','.mat','.pt','.pth','.pkl','.log'}
PREFIXES=('outputs/','reference/validation/','docs/assets/','.venv/','build/')


def generated(name):
    return name.startswith(PREFIXES) or Path(name).suffix.lower() in GENERATED


def main():
    root=Path(__file__).resolve().parents[1]
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args])
    names=git('ls-files','-z').decode().rstrip('\0').split('\0')
    issues=[];sizes=[]
    for name in names:
        p=root/name
        if not p.is_file():issues.append('Tracked file is missing: '+name);continue
        sizes.append((p.stat().st_size,name))
        if generated(name):issues.append('Generated artifact tracked: '+name)
        if p.suffix=='.ipynb':
            book=json.loads(p.read_text())
            if any(c.get('outputs') or c.get('execution_count') is not None for c in book['cells'] if c['cell_type']=='code'):
                issues.append('Notebook contains outputs: '+name)
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                if target.startswith(('https://','http://','#','mailto:')):continue
                target=target.split('#')[0]
                if not (p.parent/target).exists():issues.append(f'Broken local link: {name} -> {target}')
    history=git('rev-list','--objects','HEAD').decode().splitlines()
    historical_generated=sorted({line.partition(' ')[2] for line in history if ' ' in line and generated(line.partition(' ')[2])})
    if historical_generated:issues.append('Generated artifacts reachable in upload history: '+', '.join(historical_generated))
    archive=git('archive','--format=zip','HEAD')
    report=dict(tracked_file_count=len(names),tracked_source_bytes=sum(n for n,_ in sizes),
        head_zip_bytes=len(archive),commits_reachable_from_head=int(git('rev-list','--count','HEAD')),
        largest_tracked_files=[dict(path=p,bytes=n) for n,p in sorted(sizes,reverse=True)[:5]],issues=issues)
    print(json.dumps(report,indent=2))
    if issues:raise SystemExit(1)


if __name__=='__main__':main()
