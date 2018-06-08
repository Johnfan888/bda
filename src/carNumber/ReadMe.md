１.brand_...是按汽车品牌分别处理后，合并品牌结果
	brand_data_choose:原始数据处理
	brand_cnn:使用cnn预测，合并结果
	brand_cnn_method:CNN模块
	brand_cnn_lgb：将CNN提取的特征nn_train,nn_test与原始特征合并后使用LGB预测

２.将所有汽车品牌的上牌量合并起来，预测
	lgb_compare:  原始特征使用LGB预测
	CNN_feature:  CNN提取特征模型	
	cnn_compare:  使用CNN提取的特征预测
	cnn_lgb_compare:　　使用CNN特征+原始特征，预测
