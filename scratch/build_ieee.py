
import re, unicodedata, os

def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

with open('results/FINAL_MANUSCRIPT.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Author-year to IEEE key mapping
key_to_ieee = {
    'foley2019sex': 1,
    'meiklejohn2013fistful': 2,
    'moser2013inquiry': 3,
    'harlev2018breaking': 4,
    'hu2019transaction': 5,
    'reid2013analysis': 6,
    'he2009learning': 7,
    'weber2019elliptic': 8,
    'gama2014survey': 9,
    'quinonero2009dataset': 10,
    'bronstein2017geometric': 11,
    'wu2020comprehensive': 12,
    'kipf2017semi': 13,
    'hamilton2017inductive': 14,
    'velickovic2018graph': 15,
    'grinsztajn2022why': 16,
    'shwartz2022tabular': 17,
    'errica2020fair': 18,
    'arp2022dos': 19,
    'davis2006relationship': 20,
    'saito2015precision': 21,
    'breiman2001random': 22,
    'chen2016xgboost': 23,
    'holm1979simple': 24,
    'wilcoxon1945individual': 25,
    'demsar2006statistical': 26,
    'cohen1988statistical': 27,
    'lakens2013calculating': 28,
    'li2018deeper': 29,
    'oono2020graph': 30,
    'chen2020measuring': 31,
    'wang2021graph': 32,
    'xu2020inductive': 33,
    'rossi2020temporal': 34
}

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

print('Conversion script template ready')
