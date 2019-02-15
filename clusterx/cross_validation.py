# Copyright (c) 2015-2019, CELL Developers.
# This work is licensed under the terms of the Apache 2.0 license
# See accompanying license for details or visit https://www.apache.org/licenses/LICENSE-2.0.txt.



def loo_cv_num(fitter, data, target):
    from sklearn import linear_model
    from sklearn.model_selection import cross_val_score

    
    
