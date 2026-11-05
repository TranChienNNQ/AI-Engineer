"""Recreate the seven teaching figures. Data panels come from a fresh deterministic lab report."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import ListedColormap
from lab import report

ROOT=Path(__file__).resolve().parent
BLUE='#2863a5'; GREEN='#227b62'; RED='#bd4949'; GRAY='#596777'; GOLD='#d89a24'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'#ffffff'})


def canvas(title,subtitle,size=(13,7)):
    fig,ax=plt.subplots(figsize=size)
    ax.set(xlim=(0,13),ylim=(0,7));ax.axis('off')
    fig.suptitle(title,x=.06,y=.96,ha='left',fontsize=19,fontweight='bold',color='#172d45')
    fig.text(.06,.89,subtitle,fontsize=10,color=GRAY)
    fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.84)
    return fig,ax


def box(ax,x,y,w,h,label,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.12',
                              facecolor=color+'12',edgecolor=color,linewidth=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11,color='#172d45',linespacing=1.6)


def arrow(ax,x,y,xx,yy,color=GRAY):
    ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=15,color=color,lw=1.6))


def save(fig,name,out):
    for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=170,facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path)
    parser.add_argument('--out',type=Path,default=ROOT/'figures');args=parser.parse_args()
    r=json.loads(args.report.read_text()) if args.report else report()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    fig,ax=canvas('01 · Context đi qua các ranh giới tin cậy',
                  'Sơ đồ kiến trúc minh họa · quyền đến từ server; văn bản truy xuất chỉ là dữ liệu.')
    box(ax,.3,4.6,3.2,1.4,'CONTROL DO SERVER GIỮ\nSession → subject / tenant\nPolicy + schema + ý định hiện tại',GREEN)
    box(ax,.3,1.2,3.2,2.2,'DỮ LIỆU CÓ THỂ CHỨA INJECTION\nTài liệu · lịch sử · memory\nKết quả tool\nKhông được tự cấp role / grant',GOLD)
    box(ax,4.5,2.5,2.5,2,'CONTEXT BUILDER\nLọc quyền → xếp hạng\nLineage → budget',BLUE)
    box(ax,7.8,2.5,1.8,2,'MODEL\nĐọc context\nĐề xuất action',BLUE)
    box(ax,10.3,2.5,2.3,2,'GATEWAY\nKiểm quyền sống\nGrant + version\nRồi mới thực thi',GREEN)
    arrow(ax,3.6,5.2,5.5,4.6,GREEN);arrow(ax,3.6,2.3,4.4,3.2,GOLD)
    arrow(ax,7.1,3.5,7.7,3.5);arrow(ax,9.7,3.5,10.2,3.5)
    ax.text(8.7,1.4,'Proposal vẫn là dữ liệu không tin cậy',ha='center',color=RED)
    ax.text(6.5,.3,'Được đọc ≠ được tin làm chỉ dẫn. Được đề xuất ≠ được phép thực thi.',ha='center',fontsize=12)
    save(fig,'01-trust-boundaries',out)

    fig,ax=canvas('02 · Chọn context trong ngân sách hữu hạn',
                  'Ví dụ tính tay · đơn vị token minh họa; không phải tokenizer của nhà cung cấp.')
    segments=[('Control',128,BLUE),('Tool',96,GREEN),('Câu hỏi',32,GRAY),('Context',640,GOLD),('Output',128,RED)]
    x=.4
    for label,n,color in segments:
        width=n/1024*12.2
        ax.barh(5.5,width,left=x,height=.8,color=color)
        ax.text(x+width/2,5.5,str(n),ha='center',va='center',color='white',fontsize=10)
        x+=width
    ax.text(6.5,6.25,'Window 1.024 = 128 + 96 + 32 + 640 + 128',ha='center',fontsize=13)
    ax.text(6.5,4.55,'Rules 128   ·   Tool schema 96   ·   Câu hỏi 32   ·   Context 640   ·   Output 128',ha='center',fontsize=10)
    box(ax,.5,1.9,3.4,1.6,'BƯỚC 1: LỌC QUYỀN\nE: cost 80, utility 100\nKhông có quyền → loại cứng',RED)
    box(ax,4.8,1.9,3.4,1.6,'BƯỚC 2: CHỌN TRONG 640\nA 240/9 · B 200/8\nC 160/6 · D 120/5',BLUE)
    box(ax,9.1,1.9,3.4,1.6,'TỐI ƯU VÍ DỤ NHỎ\nA + B + C = 600\nUtility = 23',GREEN)
    arrow(ax,4,2.7,4.7,2.7);arrow(ax,8.3,2.7,9,2.7)
    ax.text(6.5,.6,'Lab dùng greedy whole-item; ví dụ knapsack này được duyệt hết tổ hợp để học.',ha='center',color=GRAY)
    save(fig,'02-context-budget',out)

    rows=r['permissions'];ids=['GLOBAL','B-PUBLIC','B-INTERNAL','G-INTERNAL','B-POISON','T-BLUE-1','T-BLUE-2','T-GREEN-1']
    matrix=np.array([[int(x in p['readable']) for x in ids] for p in rows])
    fig,ax=plt.subplots(figsize=(13,6.5));fig.subplots_adjust(left=.16,right=.96,bottom=.24,top=.77)
    ax.imshow(matrix,cmap=ListedColormap(['#f5dddd','#d9eee6']),vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(ids)),ids,rotation=25,ha='right');ax.set_yticks(range(len(rows)),[p['subject'] for p in rows])
    for i in range(len(rows)):
        for j in range(len(ids)):ax.text(j,i,'ĐỌC' if matrix[i,j] else 'CHẶN',ha='center',va='center',fontsize=10,color=GREEN if matrix[i,j] else RED)
    ax.set_xticks(np.arange(-.5,len(ids),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=3);ax.tick_params(which='minor',bottom=False,left=False)
    fig.suptitle('03 · Quyền đọc: subject × resource',x=.06,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.06,.86,f"Dữ liệu lab: {matrix.sum()}/40 cặp được đọc · tenant, scope, role, clearance và deny cùng quyết định.",fontsize=11,color=GRAY)
    fig.text(.06,.06,'Ma trận này chỉ là quyền đọc. Đóng ticket còn cần operator, scope và grant cho đúng hành động.',fontsize=11)
    save(fig,'03-permission-matrix',out)

    fig,ax=canvas('04 · Injection có thể đến từ nhiều đường',
                  'Các payload/proposal được lập trình sẵn; chưa đo hành vi của model sinh.')
    for y,label in [(5.3,'Câu hỏi trực tiếp\n“Tôi là operator”'),(3.5,'Tài liệu / tool result\n“Đổi ticket đích”'),(1.7,'Memory / summary\n“Đã đồng ý mọi thao tác”')]:
        box(ax,.4,y-.4,3.2,1.2,label,GOLD);arrow(ax,3.7,y+.2,5,3.7,GOLD)
    box(ax,5.1,2.9,2.6,1.6,'MODEL / FIXTURE\nTạo proposal\nhoặc câu trả lời',BLUE)
    arrow(ax,7.8,4,9,4.9);box(ax,9.1,4.3,3.3,1.6,'ACTION GATEWAY\nRole thật + target + grant\nChặn action ngoài quyền',GREEN)
    arrow(ax,7.8,3.2,9,2.2,RED);box(ax,9.1,1.4,3.3,1.6,'CÂU TRẢ LỜI SAI\nBANANA_OVERRIDE\nKhông có tool call → residual',RED)
    ax.text(6.5,.4,'Chặn side effect không chứng minh nội dung trả lời đúng hoặc không bị thao túng.',ha='center',fontsize=12)
    save(fig,'04-injection-paths',out)

    fig,ax=canvas('05 · Summary và cache phải giữ nguồn gốc',
                  'Dữ liệu dẫn xuất kế thừa ràng buộc truy cập của mọi nguồn đã dùng.')
    box(ax,.4,4.4,3.2,1.4,'Nguồn A\nBlue public · version 1',BLUE)
    box(ax,.4,1.7,3.2,1.4,'Nguồn B\nGreen private · version 1',RED)
    arrow(ax,3.7,5.1,5,3.9);arrow(ax,3.7,2.4,5,3.4,RED)
    box(ax,5.1,2.7,3,2,'SUMMARY\nOwner = Alice chưa đủ\nOrigins = [A, B]\nReaders ⊆ Readers(A) ∩ Readers(B)',GOLD)
    arrow(ax,8.2,3.7,9,3.7);box(ax,9.1,2.7,3.3,2,'CACHE HIT\nKiểm subject + revisions\nKiểm lại từng origin\nThu hồi / đổi version → loại',GREEN)
    ax.text(6.5,1.2,'Alice không đọc được B → không đưa summary A+B vào context của Alice.',ha='center',color=RED)
    ax.text(6.5,.4,'Epoch giúp invalidation; kiểm quyền sống là lớp dự phòng khi sự kiện invalidation bị bỏ sót.',ha='center',fontsize=11)
    save(fig,'05-cache-lineage',out)

    fig,ax=canvas('06 · Grant gắn với hành động đã review',
                  'Grant được giữ ngoài prompt · token dùng một lần · lock chỉ trong một process của lab.')
    box(ax,.4,4.5,3.3,1.5,'TRUSTED CONTROLLER\nReview đúng ticket / reason\nSinh grant bằng secrets',GREEN)
    box(ax,4.8,4.2,3.4,2.1,'GRANT STORE\nSubject + tenant + revision\nHash proposal + resource version\nACL epoch + expiry + used',BLUE)
    box(ax,9.3,4.5,3.2,1.5,'MODEL PROPOSAL\nKhông tự cấp grant\nĐổi reason → hash khác',GOLD)
    arrow(ax,3.8,5.2,4.7,5.2,GREEN);arrow(ax,6.5,4.1,6.5,3.2);arrow(ax,10.9,4.4,8.2,2.7,GOLD)
    box(ax,4.6,1.5,4.1,1.7,'MỘT VÙNG ATOMIC\nKiểm quyền sống + now < expiry\nKiểm binding → consume → mutate',GREEN)
    ax.text(2.2,2.4,'Thiếu / hết hạn / replay\n→ từ chối',ha='center',va='center',color=RED)
    ax.text(10.8,2.4,'Hợp lệ lần đầu\n→ đóng đúng ticket',ha='center',va='center',color=GREEN)
    arrow(ax,4.5,2.4,3.8,2.4,RED);arrow(ax,8.8,2.4,9.5,2.4,GREEN)
    ax.text(6.5,.4,'Cấp lúc 100, TTL 30 → expires_at 130. now=129 hợp lệ; now=130 đã hết hạn.',ha='center',fontsize=12)
    save(fig,'06-grant-gateway',out)

    a=r['actions'];c=r['context_checks']
    fig,axes=plt.subplots(1,3,figsize=(14,6.5));fig.subplots_adjust(left=.07,right=.97,bottom=.27,top=.73,wspace=.4)
    panels=[('Action ngoài quyền / ý định',[a['unauthorized_events_naive'],a['unauthorized_events_guarded']],a['attack_cases'],['Schema-only','Gateway']),
            ('Task hợp lệ hoàn thành',[a['benign_cases'],a['benign_completed']],a['benign_cases'],['Schema-only','Gateway']),
            ('Context lộ dữ liệu',[c['unsafe_exposures'],c['guarded_exposures']],c['confidentiality_cases'],['Unsafe','Lọc + recheck'])]
    for ax,(title,values,n,labels) in zip(axes,panels):
        ax.bar(labels,values,color=[GOLD,GREEN],width=.6);ax.set_ylim(0,n*1.24);ax.set_yticks(range(0,n+1,max(1,n//4)))
        ax.set_title(title,fontsize=12);ax.set_ylabel('Số ca');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(i,v+.25,f'{v}/{n}',ha='center',fontsize=13,fontweight='bold')
    fig.suptitle('07 · Đo riêng an toàn, utility và lộ context',x=.06,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.06,.85,'Kết quả chạy code cục bộ · proposal cố định và dữ liệu tổng hợp · không phải benchmark LLM.',fontsize=11,color=GRAY)
    fig.text(.06,.13,'Residual: answer-only manipulation vẫn sai; gateway không kiểm chứng ngữ nghĩa câu trả lời.',fontsize=12,color=RED)
    fig.text(.06,.06,'0/17 chỉ mô tả các fixture đã chạy; không là xác suất rủi ro 0 trong môi trường thực.',fontsize=11,color=GRAY)
    save(fig,'07-evaluation',out)
    print(json.dumps({'figures':7,'formats':['png','svg'],'dataset_hash':r['dataset_hash']},indent=2))

if __name__=='__main__':main()
