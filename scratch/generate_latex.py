# -*- coding: utf-8 -*-
"""
Script to generate results/latex/main.tex in pristine IEEEtran format.
"""

import re, os

# Map author-year strings to BibTeX keys
author_year_to_key = {
    'arp et al., 2022': 'arp2022dos',
    'breiman, 2001': 'breiman2001random',
    'bronstein et al., 2017': 'bronstein2017geometric',
    'chen and guestrin, 2016': 'chen2016xgboost',
    'chen et al., 2020': 'chen2020measuring',
    'cohen, 1988': 'cohen1988statistical',
    'davis and goadrich, 2006': 'davis2006relationship',
    'demsar, 2006': 'demsar2006statistical',
    'errica et al., 2020': 'errica2020fair',
    'foley et al., 2019': 'foley2019sex',
    'gama et al., 2014': 'gama2014survey',
    'grinsztajn et al., 2022': 'grinsztajn2022why',
    'hamilton et al., 2017': 'hamilton2017inductive',
    'harlev et al., 2018': 'harlev2018breaking',
    'he and garcia, 2009': 'he2009learning',
    'holm, 1979': 'holm1979simple',
    'hu et al., 2019': 'hu2019transaction',
    'kipf and welling, 2017': 'kipf2017semi',
    'lakens, 2013': 'lakens2013calculating',
    'li et al., 2018': 'li2018deeper',
    'meiklejohn et al., 2013': 'meiklejohn2013fistful',
    'moser et al., 2013': 'moser2013inquiry',
    'oono and suzuki, 2020': 'oono2020graph',
    'quinonero-candela et al., 2009': 'quinonero2009dataset',
    'reid and harrigan, 2013': 'reid2013analysis',
    'rossi et al., 2020': 'rossi2020temporal',
    'saito and rehmsmeier, 2015': 'saito2015precision',
    'shwartz-ziv and armon, 2022': 'shwartz2022tabular',
    'velickovic et al., 2018': 'velickovic2018graph',
    'wang et al., 2021': 'wang2021graph',
    'weber et al., 2019': 'weber2019elliptic',
    'wilcoxon, 1945': 'wilcoxon1945individual',
    'wu et al., 2020': 'wu2020comprehensive',
    'xu et al., 2020': 'xu2020inductive'
}

def clean_latex(text):
    # Convert author-year citations to \cite{...}
    def repl_cite(m):
        content = m.group(1)
        if any(k in content for k in ['TABLE', 'Figure', 'd_{', 'u \\to', 'u <->', 't \\in', 'p >=', 'p <=', 'h_']):
            return m.group(0)
        chunks = content.split(';')
        keys = []
        for c in chunks:
            c_clean = c.strip().lower()
            # Normalize unicode for matching
            matched = None
            for ay, k in author_year_to_key.items():
                if ay in c_clean or k in c_clean:
                    matched = k
                    break
            if matched:
                keys.append(matched)
            else:
                return m.group(0)
        return '\\cite{' + ', '.join(keys) + '}'
    
    # Replace citations
    text = re.sub(r'\[([A-Z\u00C0-\u017F][^\]]*?\b\d{4}\b[^\]]*?)\]', repl_cite, text)
    
    # Fix percent signs not in math
    # text = re.sub(r'(?<!\\)%', r'\\%', text)
    return text

print("clean_latex function defined.")
