"""Real OTel traces, bounded local exporter, manual aggregate metrics and correlated JSON logs."""
from collections import Counter,deque
import math
import time
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider,SpanLimits
from opentelemetry.sdk.trace.export import SpanExporter,SimpleSpanProcessor,SpanExportResult
from opentelemetry.sdk.trace.sampling import ParentBased,ALWAYS_ON,ALWAYS_OFF
from opentelemetry.trace import Status,StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

PROPAGATOR=TraceContextTextMapPropagator()
SAFE_ATTRIBUTES={'app.route','app.method','app.model_version','app.feature_version',
                 'app.status_code','app.error_code','app.queue_depth'}


def attrs(values):
    """Closed keys; callers supply static categories, never request body/headers/exception text."""
    out={}
    for key,value in values.items():
        if key not in SAFE_ATTRIBUTES:continue
        if type(value) in {int,bool}:out[key]=value
        elif isinstance(value,str):out[key]=value[:64]
    return out


class LocalExporter(SpanExporter):
    def __init__(self,limit=256,fail=False):
        if type(limit) is not int or limit<1:raise ValueError('positive exporter limit required')
        self.rows=deque(maxlen=limit);self.evicted=0;self.failed_spans=0;self.fail=fail

    def export(self,spans):
        if self.fail:
            self.failed_spans+=len(spans);return SpanExportResult.FAILURE
        for span in spans:
            if len(self.rows)==self.rows.maxlen:self.evicted+=1
            self.rows.append(dict(service=span.resource.attributes.get('service.name'),name=span.name,
                trace_id=f'{span.context.trace_id:032x}',span_id=f'{span.context.span_id:016x}',
                parent_id=f'{span.parent.span_id:016x}' if span.parent else None,
                kind=span.kind.name,status=span.status.status_code.name,
                start_ns=span.start_time,end_ns=span.end_time,attributes=attrs(span.attributes or {}),
                events=[dict(name=e.name,attributes=attrs(e.attributes or {})) for e in span.events]))
        return SpanExportResult.SUCCESS

    def shutdown(self):pass


class Histogram:
    """Teaching cumulative histogram, not an OTel Metrics SDK or Prometheus exporter."""
    def __init__(self,bounds=(.001,.005,.01,.05,.1,.5,1.0)):
        if not bounds or any(not math.isfinite(x) or x<0 for x in bounds) or list(bounds)!=sorted(set(bounds)):
            raise ValueError('strictly increasing finite nonnegative bounds required')
        self.bounds=list(bounds);self.buckets=[0]*len(bounds);self.count=0;self.total=0.0

    def observe(self,value):
        if not math.isfinite(value) or value<0:raise ValueError('nonnegative finite observation required')
        self.count+=1;self.total+=value
        for i,bound in enumerate(self.bounds):
            if value<=bound:self.buckets[i]+=1

    def snapshot(self):
        return dict(bounds=self.bounds+['+Inf'],cumulative=self.buckets+[self.count],count=self.count,sum=self.total)


class Metrics:
    def __init__(self):
        self.requests=Counter();self.duration={};self.queue_wait=Histogram()

    def completed(self,service,route,status,duration):
        if service not in {'lesson32-api','lesson32-model'}:raise ValueError('unknown service category')
        route=route if route in {'/v1/predict','/score'} else 'unmatched'
        status_class=f'{status//100}xx' if 100<=status<600 else 'other'
        self.requests[(service,route,status_class)]+=1
        self.duration.setdefault((service,route),Histogram()).observe(duration)

    def snapshot(self):
        return dict(requests=[dict(service=s,route=r,status_class=c,count=n)
                             for (s,r,c),n in sorted(self.requests.items())],
                    request_duration_seconds=[dict(service=s,route=r,**h.snapshot()) for (s,r),h in sorted(self.duration.items())],
                    queue_wait_seconds=self.queue_wait.snapshot(),
                    note='All completions, independent of trace sampling; manual aggregates, no Prometheus server')


class Telemetry:
    def __init__(self,sampled=True,export_limit=256,export_fail=False):
        self.exporter=LocalExporter(export_limit,export_fail);self.metrics=Metrics();self.logs=deque(maxlen=128)
        self.providers={};self.tracers={};self.closed=False
        for name in ['lesson32-api','lesson32-model']:
            provider=TracerProvider(resource=Resource({'service.name':name}),shutdown_on_exit=False,
                sampler=ParentBased(ALWAYS_ON) if sampled else ALWAYS_OFF,
                span_limits=SpanLimits(max_attributes=16,max_events=8,max_links=4))
            provider.add_span_processor(SimpleSpanProcessor(self.exporter))
            self.providers[name]=provider;self.tracers[name]=provider.get_tracer('course.lesson32','32.0')

    def span(self,service,name,**kwargs):
        return self.tracers[service].start_as_current_span(name,record_exception=False,
                                                          set_status_on_exception=False,**kwargs)

    def log(self,service,route,status,span):
        context=span.get_span_context()
        self.logs.append(dict(timestamp_ns=time.time_ns(),event='request_completed',service=service,route=route,
            status_code=status,trace_id=f'{context.trace_id:032x}',span_id=f'{context.span_id:016x}'))

    def close(self):
        if self.closed:return
        for provider in self.providers.values():provider.force_flush();provider.shutdown()
        self.closed=True


def error(span,code):
    span.set_attribute('app.error_code',code);span.set_status(Status(StatusCode.ERROR))


def extract(headers):
    # Only Trace Context, not a global Baggage propagator. Context() avoids accidental inherited roots.
    carrier={k.lower():v for k,v in headers.items() if k.lower() in {'traceparent','tracestate'}}
    return PROPAGATOR.extract(carrier=carrier,context=Context())


def inject():
    carrier={};PROPAGATOR.inject(carrier);return carrier
