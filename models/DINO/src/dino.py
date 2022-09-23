import torch
import torch.nn as nn
from typing import List, Tuple
from src.vit import create_vit
import timm

def create_model(
    backbone: str, 
    pretrained: bool = True,
    img_size: int = None,
    num_classes: int = 0
) -> nn.Module:
    """creates model's backbone

    Args:
        backbone (str): backbone name
        pretrained (bool, optional): pretrained. Defaults to True.
        img_size (int, optional): input image size. Defaults to 224.
        num_classes (int, optional): number of output classes. Defaults to 0.

    Returns:
        nn.Module: backbone model
    """
    
    if backbone.startswith("custom_"):
        model_info=backbone.split("_")
        img_size = int(model_info[-1]) if img_size is None else img_size
        return create_vit(
            vit_base=model_info[1],
            model_size=model_info[2],
            pretrained=pretrained,
            patch_size=int(model_info[3].replace("patch", "")),
            img_size=img_size,
            num_classes=num_classes
        )
    else:
        return timm.create_model(
            model_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )

class MLP(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        hidden_dim: int = 4096, 
        proj_dim: int = 256,
        num_layers: int = 2,
        use_bn: bool = True,
        use_gelu: bool = False,
        drop_p: float = 0.,
        init_weights: bool = True
    ) -> None:
        """MLP implementation

        Args:
            in_features (int): input features size
            hidden_dim (int, optional): hidden layer features size. Defaults to 4096.
            proj_dim (int, optional): output features size (projection). Defaults to 256.
            num_layers (int, optional): number of layers in the MLP. Defaults to 3.
            use_bn (bool, optional): whether to apply BN. Defaults to True.
            use_gelu (bool, optional): whether to use GELU (True) or ReLU (False). Defaults to True.
            drop_p (float, optional): dropout prob. Defaults to 0.
            init_weights (bool, optional): if True initialize weights. Defaults to True.
        """
        super().__init__()
        
        num_layers = max(1, num_layers)
        if num_layers == 1:
            self.model = nn.Linear(in_features=in_features, out_features=proj_dim)
        else:
            # adding first layer 
            layers = [nn.Linear(in_features=in_features, out_features=hidden_dim)]
            if use_bn: layers.append(nn.BatchNorm1d(num_features=hidden_dim))
            layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
            
            # adding all the other layers
            for _ in range(num_layers-2):
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                if use_bn: layers.append(nn.BatchNorm1d(num_features=hidden_dim))
                layers.append(nn.GELU() if use_gelu else nn.ReLU(inplace=True))
                
            layers.append(nn.Linear(in_features=hidden_dim, out_features=proj_dim))
            layers.append(nn.Dropout(drop_p))
            
            self.model = nn.Sequential(*layers)
            
            if init_weights: self.apply(self._init_weights)
                
    def _init_weights(self, m):
        """init weights fn

        Args:
            m (nn.Module): torch nn Module
        """
        if isinstance(m, nn.Linear):
            from src.functions import fills_val_trunc_normal_
            fills_val_trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on input tensor x

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        
        return self.model(x)

class Encoder(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        img_size: int,
        pretrained: bool = True,
        hidden_dim: int = 4096,
        proj_dim: int = 256,
        num_layers: int = 2,
        use_bn: bool = True,
        use_gelu: bool = False,
        drop_p: float = 0.,
        init_weights: bool = True,
        dino: bool = False,
        dino_out_dim: int = 65568,
        norm_last_layer: bool = False
    ) -> None:
        """Encoder initializer

        Args:
            backbone (str): backbone architecture
            img_size (int): input image size
            pretrained (bool, optional): load pretrained weights. Defaults to True.
            hidden_dim (int, optional): MLP hidden dim. Defaults to 4096.
            proj_dim (int, optional): MLP projector output dim. Defaults to 256.
            num_layers (int, optional): MLP number of linear layers. Defaults to 2.
            use_bn (bool, optional): use batch norm in MLP. Defaults to True.
            use_gelu (bool, optional): use GELU in MLP. Defaults to False.
            drop_p (float, optional): dropout prob in MLP. Defaults to 0..
            init_weights (bool, optional): whether to init weights in MLP. Defaults to True.
            dino (bool, optional): if DINO model and so put last layer. Defaults to False.
            dino_out_dim (int, optional): DINO output dim. Defaults to 65568.
            norm_last_layer (bool, optional): whether to normalize last layer. Defaults to False.
        """
        
        super().__init__()
        
        self.dino = dino
        
        self.backbone = create_model(
            backbone=backbone,
            pretrained=pretrained,
            img_size=img_size
        )
        from src.functions import get_out_features
        backbone_out = get_out_features(backbone)
        
        self.projector = MLP(
            in_features=backbone_out,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights
        )
        
        if dino:
            self.last_layer = nn.utils.weight_norm(nn.Linear(in_features=proj_dim, out_features=dino_out_dim, bias=False))
            self.last_layer.weight_g.data.fill_(1)
            if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.backbone(x)
        x = self.projector(x)
        if hasattr(self, 'last_layer'):
            x = nn.functional.normalize(x, dim=-1, p=2)             
            x = self.last_layer(x)
        return x
        

class DINO(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        img_size: int,
        pretrained: bool = True,
        hidden_dim: int = 4096,
        proj_dim: int = 256,
        out_dim: int = 65568,
        num_layers: int = 3,
        use_bn: bool = False,
        use_gelu: bool = False,
        drop_p: float = 0.,
        init_weights: bool = True,
        norm_last_layer: bool = True,
        beta: float = 0.996,
    ) -> None:
        """DINO Model initialization.

        Args:
            backbone (str): backbone architecture
            img_size (int): input image size
            pretrained (bool, optional): load pretrained weights. Defaults to True.
            hidden_dim (int, optional): encoder hidden dim. Defaults to 4096.
            proj_dim (int, optional): encoder projector output dim. Defaults to 256.
            out_dim (int, optional): dimensionality of the DINO head output. For complex and large datasets large values (like 65k) work well.. Defaults to 65568.
            num_layers (int, optional): encoder num of linear layers. Defaults to 3.
            use_bn (bool, optional): use batch norm in encoder. Defaults to False.
            use_gelu (bool, optional): use gelu in encoder. Defaults to False.
            drop_p (float, optional): dropout in encoder. Defaults to 0..
            init_weights (bool, optional): init weights in encoder. Defaults to True.
            norm_last_layer (bool, optional): normalize last layer in encoder. Defaults to True.
            beta (float, optional): EMA update weight. Defaults to 0.996.
        """
        super().__init__()
        
        self.student = Encoder(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights,
            dino=True,
            dino_out_dim=out_dim,
            norm_last_layer=norm_last_layer
        )
        
        self.teacher = Encoder(
            backbone=backbone,
            img_size=img_size,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            num_layers=num_layers,
            use_bn=use_bn,
            use_gelu=use_gelu,
            drop_p=drop_p,
            init_weights=init_weights,
            dino=True,
            dino_out_dim=out_dim,
            norm_last_layer=norm_last_layer
        )
        
        self.beta = beta
        
        self._init_teacher_weights()
        
    def _init_teacher_weights(self):
        """inits f() weights with g() weights. Also sets the gradient of f() to False.
        """
        for params_student, params_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            params_teacher.data.copy_(params_student.data) # copying g params
            params_teacher.requires_grad = False # no gradient updates
            
    @torch.no_grad()
    def update_teacher(self):
        """EMA update of the target network
        """
        # TODO: implementare lo scheduler di beta come da paper (va da 0.996 a 1)
        for params_student, params_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            params_teacher.data = self.beta * params_teacher.data + (1 - self.beta) * params_student.data
            
    def embeds(self, x: torch.Tensor) -> torch.Tensor:
        return self.student.backbone(x)
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x_global = x[:2]
        x_local = x[2:]
        
        # Teacher Output - global crops
        teacher_crops = len(x_global)
        x_teacher = torch.cat(x_global, dim=0) # (batch_size * 2, 3, size, size)
        teacher_logits = self.teacher(x_teacher) # (batch_size * 2, proj_dim)
        
        # Student Output - local + global crops
        # global + local
        student_crops = len(x)
        x_global_student = torch.cat(x_global, dim=0)
        x_local_student = torch.cat(x_local, dim=0)
        student_global_logits = self.student(x_global_student)
        student_local_logits = self.student(x_local_student)
        student_logits = torch.cat((student_global_logits, student_local_logits), dim=0) # (batch_size * n_crops, out_dim) -- n_crops is 2+n_local_crops (2 is global)
        
        return student_logits.chunk(student_crops), teacher_logits.chunk(teacher_crops)
        