from torch import nn
import torch

class F1ScoreLoss(nn.Module):
    """
    鍙井鍒嗙殑F1-Score鎹熷け鍑芥暟锛圖ice Loss鐨勫鍒嗙被鐗堟湰锛?
    閫傜敤浜庣被鍒笉骞宠　闂
    """
    def __init__(self, num_classes=2, epsilon=1e-7, average='macro'):
        """
        Args:
            num_classes: 绫诲埆鏁伴噺
            epsilon: 闃叉闄ら浂鐨勫皬甯告暟
            average: 骞冲潎鏂瑰紡 ('macro', 'micro', 'weighted')
        """
        super(F1ScoreLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.average = average
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 妯″瀷杈撳嚭 (batch_size, num_classes)
            targets: 鐪熷疄鏍囩 (batch_size,)
        Returns:
            F1鎹熷け鍊?(1 - F1_score)
        """
        # 灏嗛娴嬭浆鎹负姒傜巼
        probs = torch.softmax(predictions, dim=1)
        
        # 灏嗙洰鏍囪浆鎹负one-hot缂栫爜
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        
        if self.average == 'macro':
            # 瀹忓钩鍧囷細璁＄畻姣忎釜绫诲埆鐨凢1锛岀劧鍚庡彇骞冲潎
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
            # 寰钩鍧囷細鍏ㄥ眬璁＄畻TP銆丗P銆丗N
            tp = (probs * targets_one_hot).sum()
            fp = (probs * (1 - targets_one_hot)).sum()
            fn = ((1 - probs) * targets_one_hot).sum()
            
            precision = tp / (tp + fp + self.epsilon)
            recall = tp / (tp + fn + self.epsilon)
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)
            
        elif self.average == 'weighted':
            # 鍔犳潈骞冲潎锛氭牴鎹被鍒牱鏈暟鍔犳潈
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
        
        # 杩斿洖1-F1浣滀负鎹熷け锛堥渶瑕佹渶灏忓寲锛?
        return 1 - f1

class F2ScoreLoss(nn.Module):
    """
    鍙井鍒嗙殑F2-Score鎹熷け鍑芥暟
    F2-score鏇村己璋冨彫鍥炵巼锛坮ecall锛夛紝閫傜敤浜庢紡妫€浠ｄ环楂樼殑鍦烘櫙
    """
    def __init__(self, num_classes=2, beta=2.0, epsilon=1e-7, average='macro'):
        """
        Args:
            num_classes: 绫诲埆鏁伴噺
            beta: F-beta鐨刡eta鍊硷紝beta=2鏃朵负F2-score
            epsilon: 闃叉闄ら浂鐨勫皬甯告暟
            average: 骞冲潎鏂瑰紡 ('macro', 'micro', 'weighted')
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
            predictions: 妯″瀷杈撳嚭 (batch_size, num_classes)
            targets: 鐪熷疄鏍囩 (batch_size,)
        Returns:
            F2鎹熷け鍊?(1 - F2_score)
        """
        # 灏嗛娴嬭浆鎹负姒傜巼
        probs = torch.softmax(predictions, dim=1)
        
        # 灏嗙洰鏍囪浆鎹负one-hot缂栫爜
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

def calculate_fbeta_score(y_true, y_pred, beta=2.0, num_classes=2, epsilon=1e-7):
    """
    璁＄畻瀹忓钩鍧嘑-beta Score
    
    F-beta = (1 + beta^2) * (Precision * Recall) / (beta^2 * Precision + Recall)
    beta=2鏃朵负F2-score锛屾洿寮鸿皟鍙洖鐜?
    
    Args:
        y_true: 鐪熷疄鏍囩鍒楄〃
        y_pred: 棰勬祴鏍囩鍒楄〃
        beta: F-beta鐨刡eta鍊?
        num_classes: 绫诲埆鏁伴噺
        epsilon: 闃叉闄ら浂鐨勫皬甯告暟
    
    Returns:
        F-beta score (float)
    """
    import numpy as np
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    beta_sq = beta ** 2
    
    fbeta_scores = []
    
    for i in range(num_classes):
        # 璁＄畻姣忎釜绫诲埆鐨凾P銆丗P銆丗N
        tp = np.sum((y_pred == i) & (y_true == i))
        fp = np.sum((y_pred == i) & (y_true != i))
        fn = np.sum((y_pred != i) & (y_true == i))
        
        # 璁＄畻绮剧‘鐜囧拰鍙洖鐜?
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # 璁＄畻F-beta
        if precision + recall > epsilon:
            fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + epsilon)
        else:
            fbeta = 0.0
        
        fbeta_scores.append(fbeta)
    
    # 瀹忓钩鍧?
    return np.mean(fbeta_scores)


def calculate_f2_score(y_true, y_pred, num_classes=2, epsilon=1e-7):
    """
    璁＄畻瀹忓钩鍧嘑2-score锛團-beta涓璪eta=2鐨勭壒渚嬶級
    
    F2鏇村己璋冨彫鍥炵巼锛圧ecall锛夛紝閫傜敤浜庢紡妫€浠ｄ环楂樼殑鍦烘櫙
    
    Args:
        y_true: 鐪熷疄鏍囩鍒楄〃
        y_pred: 棰勬祴鏍囩鍒楄〃
        num_classes: 绫诲埆鏁伴噺
        epsilon: 闃叉闄ら浂鐨勫皬甯告暟
    
    Returns:
        F2 score (float)
    """
    return calculate_fbeta_score(y_true, y_pred, beta=2.0, num_classes=num_classes, epsilon=epsilon)


class CombinedLoss(nn.Module):
    """
    缁勫悎鎹熷け鍑芥暟锛欳rossEntropy + F1/F2
    缁撳悎涓よ€呯殑浼樼偣锛欳E浼樺寲鍒嗙被杈圭晫锛孎1/F2澶勭悊绫诲埆涓嶅钩琛?
    """
    def __init__(self, num_classes=2, loss_type='f1', ce_weight=0.5, f_weight=0.5, 
                 beta=2.0, epsilon=1e-7, average='macro', class_weights=None, label_smoothing=0.0):
        """
        Args:
            num_classes: 绫诲埆鏁伴噺
            loss_type: 'f1' 鎴?'f2'
            ce_weight: CrossEntropy鎹熷け鐨勬潈閲?
            f_weight: F1/F2鎹熷け鐨勬潈閲?
            beta: F2-score鐨刡eta鍊?
            class_weights: 绫诲埆鏉冮噸
            label_smoothing: 鏍囩骞虫粦绯绘暟
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


class FocalLoss(nn.Module):
    """
    Focal Loss鎹熷け鍑芥暟
    閫氳繃闄嶄綆鏄撳垎绫绘牱鏈殑鏉冮噸锛岃仛鐒︿簬闅惧垎绫绘牱鏈紝瑙ｅ喅绫诲埆涓嶅钩琛￠棶棰?
    
    鍘熷璁烘枃: "Focal Loss for Dense Object Detection" (RetinaNet)
    """
    def __init__(self, num_classes=2, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Args:
            num_classes: 绫诲埆鏁伴噺
            alpha: 绫诲埆鏉冮噸锛屽彲浠ユ槸鏍囬噺銆佸垪琛ㄦ垨寮犻噺
                   - None: 涓嶄娇鐢ㄧ被鍒潈閲?
                   - float: 鐢ㄤ簬浜屽垎绫荤殑姝ｇ被鏉冮噸
                   - list/tensor: 姣忎釜绫诲埆鐨勬潈閲?
            gamma: 鑱氱劍鍙傛暟锛実amma瓒婂ぇ锛屽鏄撳垎绫绘牱鏈殑鎯╃綒瓒婂ぇ
                   - gamma=0: 绛変环浜嶤rossEntropy
                   - gamma=2: 甯哥敤榛樿鍊?
            reduction: 鎹熷け鑱氬悎鏂瑰紡 ('mean', 'sum', 'none')
            label_smoothing: 鏍囩骞虫粦绯绘暟
        """
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 澶勭悊alpha鍙傛暟
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            # 浜屽垎绫绘儏鍐碉細alpha涓烘绫绘潈閲?
            self.alpha = torch.tensor([1.0 - alpha, alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 妯″瀷杈撳嚭 (batch_size, num_classes)
            targets: 鐪熷疄鏍囩 (batch_size,)
        Returns:
            Focal Loss鍊?
        """
        # 鑾峰彇璁惧淇℃伅
        device = predictions.device
        
        # 灏哸lpha绉诲埌鐩稿悓璁惧
        if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(device)
        
        # 璁＄畻log姒傜巼 (浣跨敤log_softmax鏁板€兼洿绋冲畾)
        log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
        probs = torch.exp(log_probs)
        
        # 搴旂敤鏍囩骞虫粦
        if self.label_smoothing > 0:
            # 骞虫粦鍚庣殑鐩爣鍒嗗竷
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
            targets_smooth = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
            targets_smooth = targets_one_hot
        
        # 璁＄畻浜ゅ弶鐔垫崯澶?(姣忎釜鏍锋湰姣忎釜绫诲埆)
        ce_loss = -targets_smooth * log_probs  # (batch_size, num_classes)
        
        # 璁＄畻璋冨埗鍥犲瓙 (1 - p_t)^gamma
        # p_t鏄ā鍨嬪姝ｇ‘绫诲埆鐨勯娴嬫鐜?
        p_t = (targets_one_hot * probs).sum(dim=1)  # (batch_size,)
        p_t = p_t.unsqueeze(1).expand(-1, self.num_classes)  # (batch_size, num_classes)
        
        # 璋冨埗鍥犲瓙: 瀵规槗鍒嗙被鏍锋湰(p_t澶?闄嶄綆鏉冮噸
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        
        # 搴旂敤alpha鏉冮噸
        if self.alpha is not None:
            # alpha_t: 姣忎釜鏍锋湰鐨刟lpha鏉冮噸
            if self.alpha.dim() == 1 and self.alpha.size(0) == self.num_classes:
                # 澶氱被鍒潈閲?
                alpha_t = targets_one_hot * self.alpha.unsqueeze(0)
                alpha_t = alpha_t.sum(dim=1).unsqueeze(1).expand(-1, self.num_classes)
            else:
                alpha_t = 1.0
        else:
            alpha_t = 1.0
        
        # Focal Loss = alpha_t * (1 - p_t)^gamma * CE_loss
        focal_loss = alpha_t * modulating_factor * ce_loss
        
        # 鑱氬悎鎹熷け
        if self.reduction == 'mean':
            return focal_loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss.sum(dim=1)
