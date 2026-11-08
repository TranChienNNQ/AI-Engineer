"""Learn a proxy reward, then optimize an enumerated bandit. No PPO/rollouts."""
import torch
from torch.nn import functional as F
from objectives import positive


def optimal(ref,reward,beta):
    positive(beta)
    if (ref<=0).any() or not torch.isclose(ref.sum(),ref.new_tensor(1.)):raise ValueError('positive normalized reference')
    return (ref.log()+reward/beta).softmax(-1)


def run():
    # Three votes prefer inspect (feature 1) to guess (0); one reverses it.
    w=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64));opt=torch.optim.Adam([w],lr=.05)
    labels=torch.tensor([1.,1.,1.,-1.],dtype=torch.float64)
    for _ in range(500):
        opt.zero_grad();loss=-F.logsigmoid(labels*w).mean();loss.backward();opt.step()
    weight=w.detach();features=torch.tensor([1.,0.,3.],dtype=torch.float64)
    reward=features*weight;ref=torch.tensor([.5,.49,.01],dtype=torch.float64)
    utility=torch.tensor([1.,0.,-1.],dtype=torch.float64) # evaluator, never in optimization
    results=[]
    for beta in [.5,1.,5.]:
        logits=torch.nn.Parameter(ref.log().clone());optimizer=torch.optim.Adam([logits],lr=.05)
        target=optimal(ref,reward,beta)
        for _ in range(2000):
            lp=logits.log_softmax(-1);p=lp.exp();kl=(p*(lp-ref.log())).sum()
            objective=(p*reward).sum()-beta*kl
            optimizer.zero_grad();(-objective).backward();optimizer.step()
        p=logits.detach().softmax(-1);kl=(p*(p.log()-ref.log())).sum()
        results.append(dict(beta=beta,policy=p.tolist(),closed_form=target.tolist(),
                            max_error=(p-target).abs().max().item(),proxy_reward=(p*reward).sum().item(),
                            independent_utility=(p*utility).sum().item(),kl=kl.item()))
    return dict(actions=['inspect','guess','verbose_false'],features=features.tolist(),weight=weight.item(),
                observed_pair_probability=weight.sigmoid().item(),rm_vote_nll=(-F.logsigmoid(labels*weight).mean()).item(),
                reference=ref.tolist(),reference_utility=(ref*utility).sum().item(),
                reference_proxy_reward=(ref*reward).sum().item(),results=results,
                scope='synthetic votes; exact enumeration; no PPO or human annotation')
