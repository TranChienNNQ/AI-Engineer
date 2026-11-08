"""Synthetic fixtures, not training or production data."""
FEATURES = ['latency_ratio', 'error_ratio']


def datasets():
    train = [dict(id=f'train-{i}', group=f'train-g{i}', segment='web',
                  x=[x, 0.0], y=int(x >= .5))
             for i, x in enumerate([.05,.15,.25,.35,.45,.55,.65,.75,.85,.95])]
    points = [('mobile',.3,0),('mobile',.7,1),('web',.1,0),('web',.2,0),
              ('web',.6,1),('web',.65,1),('web',.75,1),('web',.8,1),('web',.9,1),('web',.95,1)]
    validation = [dict(id=f'validation-{i}',group=f'validation-g{i}',segment=s,x=[x,0.0],y=y)
                  for i,(s,x,y) in enumerate(points)]
    return dict(schema='ticket-data-v1',features=FEATURES,rows=train), dict(schema='ticket-data-v1',features=FEATURES,rows=validation)


def validate_data(train, validation):
    for dataset in [train,validation]:
        if set(dataset) != {'schema','features','rows'} or dataset['schema']!='ticket-data-v1' or dataset['features']!=FEATURES:
            raise ValueError('dataset_contract')
        if not dataset['rows']:raise ValueError('empty_dataset')
        ids=[]
        for row in dataset['rows']:
            if set(row)!={'id','group','segment','x','y'}:raise ValueError('row_contract')
            if not all(isinstance(row[k],str) and row[k] for k in ['id','group']):raise ValueError('row_identity')
            if row['segment'] not in {'mobile','web'}:raise ValueError('segment')
            if type(row['y']) is not int or row['y'] not in {0,1}:raise ValueError('label')
            from model import valid_vector
            valid_vector(row['x']);ids.append(row['id'])
        if len(set(ids))!=len(ids):raise ValueError('duplicate_id')
    for key in ['id','group']:
        if {r[key] for r in train['rows']} & {r[key] for r in validation['rows']}:
            raise ValueError('split_overlap_'+key)
