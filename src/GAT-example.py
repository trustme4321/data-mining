import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.optim.lr_scheduler import StepLR

# 노드 특정 행렬 (6차원 특징을 가진 4개 노드)
x = torch.tensor([
    [101, 501, 3001, 2, 10, 1],
    [102, 502, 3002, 3, 12, 1],
    [103, 503, 3003, 1, 8, 2],
    [104, 504, 3004, 4, 15, 2]
], dtype=torch.float)

# 인접 행렬 (edge list 형식)
edge_index = torch.tensor([
    [0, 1, 2, 3, 0, 1],
    [1, 0, 3, 2, 2, 3]
], dtype=torch.long)

# 데이터 객체 생성
data = Data(x=x, edge_index=edge_index)

# GAT 모델 정의
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, 8, heads=heads, dropout=dropout)
        self.gat2 = GATConv(8 * heads, out_channels, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# 입력과 출력 차원 설정
in_channels =x.shape[1] # 6차원 입력
out_channels = 2 # 예제에서는 2차원 임베딩

# 모델 생성
model = GAT(in_channels, out_channels, heads=4)

# 임의의 target 생성 (예: 노드 분류)
targets = torch.tensor([0,1,0,1],dtype=torch.long)

# 손실 함수 및 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5) # 학습률 스케줄러 추가

# 모델 학습
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()
    scheduler.step() # 학습률 스케쥴러 업데이트
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 모델 평가 
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
accuracy = (pred == targets).sum().item() / targets.size(0)
print(f'Accuracy: {accuracy}')

