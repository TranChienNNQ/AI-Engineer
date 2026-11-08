import math,re,unicodedata
from collections import Counter
from corpus import digest
STOP={'và','là','cho','của','tôi','có','cách','cần','làm','thế','nào','với','xin','giúp','hãy','một'}


def tokens(text):return [w for w in re.findall(r'\w+',unicodedata.normalize('NFC',text).casefold()) if len(w)>1 and w not in STOP]


def eligible(doc,principal,service,revoked):
    return doc['tenant']==principal['tenant'] and doc['service']==service and principal['role'] in doc['roles'] and doc['active'] and doc['approved'] and doc['id'] not in revoked


def retrieve(docs,principal,service,query,min_terms=2,context_chars=600,revoked=frozenset()):
    # Both the candidates and IDF statistics are computed only after authorization filtering.
    allowed=[d for d in docs if eligible(d,principal,service,revoked)];query=set(tokens(query));counts=[Counter(tokens(d['text'])) for d in allowed]
    if not counts or not query:return [],dict(eligible=len(allowed),scored=0,packed_chars=0)
    lengths=[sum(c.values()) for c in counts];avg=sum(lengths)/len(lengths);ranked=[]
    for d,c,length in zip(allowed,counts,lengths):
        matched=query&c.keys()
        if len(matched)<min_terms:continue
        value=0.
        for term in sorted(query):
            df=sum(term in other for other in counts);idf=math.log(1+(len(counts)-df+.5)/(df+.5));tf=c[term]
            value+=idf*tf*2.2/(tf+1.2*(.25+.75*length/avg)) if tf else 0.
        ranked.append((value,d))
    ranked.sort(key=lambda pair:(-pair[0],pair[1]['id'],pair[1]['version']))
    evidence=[];remaining=context_chars
    for value,d in ranked[:2]:
        if len(d['text'])>remaining:continue  # Whole paragraph; no dangling/truncated quote.
        evidence.append(dict(doc_id=d['id'],version=d['version'],start=0,end=len(d['text']),quote=d['text'],content_hash=digest(d['text'])))
        remaining-=len(d['text'])
    return evidence,dict(eligible=len(allowed),scored=len(counts),packed_chars=context_chars-remaining)


def validate(citations,evidence,docs,principal,service,revoked=frozenset()):
    if not citations:return False
    for cite in citations:
        if cite not in evidence:return False
        matches=[d for d in docs if d['id']==cite['doc_id'] and d['version']==cite['version'] and eligible(d,principal,service,revoked)]
        if len(matches)!=1:return False
        d=matches[0]
        if type(cite['start']) is not int or type(cite['end']) is not int or not 0<=cite['start']<cite['end']<=len(d['text']):return False
        if cite['quote']!=d['text'][cite['start']:cite['end']] or cite['content_hash']!=digest(d['text']):return False
    return True
