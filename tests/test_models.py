import torch  
import torch.optim as optim

from argparse import Namespace 
from pathlib import Path 

from recsys import config, models, utils, config, data

class Testmfpt:
    def setup_method(self):
        self.n_users = 944 
        self.n_items = 1683 
        self.n_factors = 20 
        self.dropout_p = 1e-3 
        self.device = torch.device('cpu')
        self.params_fp = str(Path(config.config_dir, "test_params.json"))  
    
        self.user = torch.tensor([1,2])
        self.item = torch.tensor([1,2])
        params = Namespace(
            n_users = self.n_users,
            n_items = self.n_items,
            n_factors = self.n_factors,
            dropout_p = self.dropout_p 
        )

        utils.set_seed()
        self.mfpt = models.initialize_model(
            n_users =  int(self.n_users),
            n_items = int(self.n_items),
            params_fp = str(Path(config.config_dir, "test_params.json")),
            dropout_p = torch.device('cpu')
        )
      
    def teardown_method(self):
        del self.mfpt  
    
    def test_initialize_model(self):
        utils.set_seed()
        model = models.mfpt(
            n_users = self.n_users,
            n_items = self.n_items,
            n_factors = self.n_factors,
            dropout_p = self.dropout_p
        )

        for params1 , params2 in zip(model.parameters(), self.mfpt.parameters()):
            assert not params1.data.ne(params2.data).sum() > 0
        assert self.mpft.n_factors == model.n_factors 
        
    def test_init(self):
        assert tuple(self.mfpt.user_factors(self.user).shape) == (self.user.shape[0], self.n_factors)
        assert tuple(self.mfpt.item_factors(self.item).shape) == (self.item.shape[0], self.n_factors)

        assert tuple(self.mfpt.users_biases(self.user).shape) == (self.user.shape[0], 1)
        assert tuple(self.mfpt.item_biases(self.item).shape) == (self.item.shape[0], 1)

    def test_forward(self):
        assert len(self.user) == len(self.item)
        z = self.mfpt.forward(self.user, self.item)
        assert len(z) == 2 
