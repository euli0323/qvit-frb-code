from torch import nn
import torch

'可微分的F1-Score损失函数'
class F1ScoreLoss(nn.Module):
    """
    可微分的F1-Score损失函数（Dice Loss的多分类版本）
    适用于类别不平衡问题
    """
    def __init__(self, num_classes=2, epsilon=1e-7, average='macro'):
        """
        Args:
            num_classes: 类别数量
            epsilon: 防止除零的小常数
            average: 平均方式 ('macro', 'micro', 'weighted')
        """
        super(F1ScoreLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.average = average
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        Returns:
            F1损失值 (1 - F1_score)
        """
        # 将预测转换为概率
        probs = torch.softmax(predictions, dim=1)
        
        # 将目标转换为one-hot编码
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        
        if self.average == 'macro':
            # 宏平均：计算每个类别的F1，然后取平均
            f1_scores = []
            for i in range(self.num_classes):
                pred_i = probs[:, i]
                target_i = targets_one_hot[:, i]
                
                tp = (pred_i * target_i).sum()
                fp = (pred_i * (1 - target_i)).sum()
                fn = ((1 - pred_i) * target_i).sum()
                
                precision = tp / (tp + fp + self.epsilon)
                recall = tp / (tp + fn + self.epsilon)
                f1 = 2 * precision * recall / (precision + recall + self.epsilon)
                f1_scores.append(f1)
            
            f1 = torch.stack(f1_scores).mean()
            
        elif self.average == 'micro':
            # 微平均：全局计算TP、FP、FN
            tp = (probs * targets_one_hot).sum()
            fp = (probs * (1 - targets_one_hot)).sum()
            fn = ((1 - probs) * targets_one_hot).sum()
            
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)
            
        elif self.average == 'weighted':
            # 加权平均：根据类别样本数加权
            f1_scores = []
            weights = []
            for i in range(self.num_classes):
                pred_i = probs[:, i]
                target_i = targets_one_hot[:, i]
                
                tp = (pred_i * target_i).sum()
                fp = (pred_i * (1 - target_i)).sum()
                fn = ((1 - pred_i) * target_i).sum()
                
                precision = tp / (tp + fp + self.epsilon)
                recall = tp / (tp + fn + self.epsilon)
                f1 = 2 * precision * recall / (precision + recall + self.epsilon)
                f1_scores.append(f1)
                weights.append(target_i.sum())
            
            weights = torch.stack(weights)
            weights = weights / weights.sum()
            f1 = (torch.stack(f1_scores) * weights).sum()
        
        # 返回1-F1作为损失（需要最小化）
        return 1 - f1

'可微分的F2-Score损失函数'
class F2ScoreLoss(nn.Module):
    """
    可微分的F2-Score损失函数
    F2-score更强调召回率（recall），适用于漏检代价高的场景
    """
    def __init__(self, num_classes=2, beta=2.0, epsilon=1e-7, average='macro'):
        """
        Args:
            num_classes: 类别数量
            beta: F-beta的beta值，beta=2时为F2-score
            epsilon: 防止除零的小常数
            average: 平均方式 ('macro', 'micro', 'weighted')
        """
        super(F2ScoreLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.beta_sq = beta ** 2
        self.epsilon = epsilon
        self.average = average
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        Returns:
            F2损失值 (1 - F2_score)
        """
        # 将预测转换为概率
        probs = torch.softmax(predictions, dim=1)
        
        # 将目标转换为one-hot编码
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        
        if self.average == 'macro':
            f2_scores = []
            for i in range(self.num_classes):
                pred_i = probs[:, i]
                target_i = targets_one_hot[:, i]
                
                tp = (pred_i * target_i).sum()
                fp = (pred_i * (1 - target_i)).sum()
                fn = ((1 - pred_i) * target_i).sum()
                
                precision = tp / (tp + fp + self.epsilon)
                recall = tp / (tp + fn + self.epsilon)
                
                # F2 = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
                f2 = (1 + self.beta_sq) * precision * recall / (self.beta_sq * precision + recall + self.epsilon)
                f2_scores.append(f2)
            
            f2 = torch.stack(f2_scores).mean()
            
        elif self.average == 'micro':
            tp = (probs * targets_one_hot).sum()
            fp = (probs * (1 - targets_one_hot)).sum()
            fn = ((1 - probs) * targets_one_hot).sum()
            
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            
            f2 = (1 + self.beta_sq) * precision * recall / (self.beta_sq * precision + recall + self.epsilon)
            
        elif self.average == 'weighted':
            f2_scores = []
            weights = []
            for i in range(self.num_classes):
                pred_i = probs[:, i]
                target_i = targets_one_hot[:, i]
                
                tp = (pred_i * target_i).sum()
                fp = (pred_i * (1 - target_i)).sum()
                fn = ((1 - pred_i) * target_i).sum()
                
                precision = tp / (tp + fp + self.epsilon)
                recall = tp / (tp + fn + self.epsilon)
                
                f2 = (1 + self.beta_sq) * precision * recall / (self.beta_sq * precision + recall + self.epsilon)
                f2_scores.append(f2)
                weights.append(target_i.sum())
            
            weights = torch.stack(weights)
            weights = weights / weights.sum()
            f2 = (torch.stack(f2_scores) * weights).sum()
        
        return 1 - f2

'计算F-beta Score（用于验证阶段的模型选择）'
def calculate_fbeta_score(y_true, y_pred, beta=2.0, num_classes=2, epsilon=1e-7):
    """
    计算宏平均F-beta Score
    
    F-beta = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
    beta=2时为F2-score，更强调召回率
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        beta: F-beta的beta值
        num_classes: 类别数量
        epsilon: 防止除零的小常数
    
    Returns:
        F-beta score (float)
    """
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    beta_sq = beta ** 2
    
    fbeta_scores = []
    
    for i in range(num_classes):
        # 计算每个类别的TP、FP、FN
        tp = np.sum((y_pred == i) & (y_true == i))
        fp = np.sum((y_pred == i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))
        
        # 计算精确率和召回率
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # 计算F-beta
        if precision + recall > epsilon:
            fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
        else:
            fbeta = 0.0
        
        fbeta_scores.append(fbeta)
    
    # 宏平均
    return np.mean(fbeta_scores)


def calculate_f2_score(y_true, y_pred, num_classes=2, epsilon=1e-7):
    """
    计算宏平均F2-score（F-beta中beta=2的特例）
    
    F2更强调召回率（Recall），适用于漏检代价高的场景
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        num_classes: 类别数量
        epsilon: 防止除零的小常数
    
    Returns:
        F2 score (float)
    """
    return calculate_fbeta_score(y_true, y_pred, beta=2.0, num_classes=num_classes, epsilon=epsilon)


'组合损失函数：CrossEntropy + F1/F2'
class CombinedLoss(nn.Module):
    """
    组合损失函数：CrossEntropy + F1/F2
    结合两者的优点：CE优化分类边界，F1/F2处理类别不平衡
    """
    def __init__(self, num_classes=2, loss_type='f1', ce_weight=0.5, f_weight=0.5, 
                 beta=2.0, epsilon=1e-7, average='macro', class_weights=None, label_smoothing=0.0):
        """
        Args:
            num_classes: 类别数量
            loss_type: 'f1' 或 'f2'
            ce_weight: CrossEntropy损失的权重
            f_weight: F1/F2损失的权重
            beta: F2-score的beta值
            class_weights: 类别权重
            label_smoothing: 标签平滑系数
        """
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.f_weight = f_weight
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        
        if loss_type == 'f1':
            self.f_loss = F1ScoreLoss(num_classes, epsilon, average)
        else:  # f2
            self.f_loss = F2ScoreLoss(num_classes, beta, epsilon, average)
    
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        f = self.f_loss(predictions, targets)
        return self.ce_weight * ce + self.f_weight * f


'Focal Loss损失函数'
class FocalLoss(nn.Module):
    """
    Focal Loss损失函数
    通过降低易分类样本的权重，聚焦于难分类样本，解决类别不平衡问题
    
    原始论文: "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    def __init__(self, num_classes=2, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Args:
            num_classes: 类别数量
            alpha: 类别权重，可以是标量、列表或张量
                   - None: 不使用类别权重
                   - float: 用于二分类的正类权重
                   - list/tensor: 每个类别的权重
            gamma: 聚焦参数，gamma越大，对易分类样本的惩罚越大
                   - gamma=0: 等价于CrossEntropy
                   - gamma=2: 常用默认值
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
            label_smoothing: 标签平滑系数
        """
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 处理alpha参数
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            # 二分类情况：alpha为正类权重
            self.alpha = torch.tensor([1.0 - alpha, alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出 (batch_size, num_classes)
            targets: 真实标签 (batch_size,)
        Returns:
            Focal Loss值
        """
        # 获取设备信息
        device = predictions.device
        
        # 将alpha移到相同设备
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(device)
        
        # 计算log概率 (使用log_softmax数值更稳定)
        log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
        probs = torch.exp(log_probs)
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            # 平滑后的目标分布
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
            targets_smooth = targets_one_hot
        
        # 计算交叉熵损失 (每个样本每个类别)
        ce_loss = -targets_smooth * log_probs  # (batch_size, num_classes)
        
        # 计算调制因子 (1 - p_t)^gamma
        # p_t是模型对正确类别的预测概率
        p_t = (targets_one_hot * probs).sum(dim=1)  # (batch_size,)
        p_t = p_t.unsqueeze(1).expand(-1, self.num_classes)  # (batch_size, num_classes)
        
        # 调制因子: 对易分类样本(p_t大)降低权重
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        
        # 应用alpha权重
        if self.alpha is not None:
            # alpha_t: 每个样本的alpha权重
            if self.alpha.dim() == 1 and self.alpha.size(0) == self.num_classes:
                # 多类别权重
                alpha_t = targets_one_hot * self.alpha.unsqueeze(0)
                alpha_t = alpha_t.sum(dim=1).unsqueeze(1).expand(-1, self.num_classes)
            else:
                alpha_t = 1.0
        else:
            alpha_t = 1.0
        
        # Focal Loss = alpha_t * (1 - p_t)^gamma * CE_loss
        focal_loss = alpha_t * modulating_factor * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss.sum(dim=1)