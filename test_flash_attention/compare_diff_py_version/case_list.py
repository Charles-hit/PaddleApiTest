
# shape_q, shape_k, shape_v, dropout, causal, return_softmax
case_list = [
    (  [1, 1059, 32, 64] ,  [1, 1059, 32, 64] ,  [1, 1059, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1047, 32, 64] ,  [1, 1047, 32, 64] ,  [1, 1047, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1064, 32, 64] ,  [1, 1064, 32, 64] ,  [1, 1064, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1043, 32, 64] ,  [1, 1043, 32, 64] ,  [1, 1043, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1025, 40, 128] ,  [1, 1025, 40, 128] ,  [1, 1025, 40, 128] ,  0.0,  False,  False,  ), 
    (  [1, 1054, 32, 64] ,  [1, 1054, 32, 64] ,  [1, 1054, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1099, 32, 64] ,  [1, 1099, 32, 64] ,  [1, 1099, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1045, 32, 64] ,  [1, 1045, 32, 64] ,  [1, 1045, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1071, 32, 64] ,  [1, 1071, 32, 64] ,  [1, 1071, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1379, 32, 64] ,  [1, 1379, 32, 64] ,  [1, 1379, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1038, 32, 64] ,  [1, 1038, 32, 64] ,  [1, 1038, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1040, 32, 64] ,  [1, 1040, 32, 64] ,  [1, 1040, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1058, 32, 64] ,  [1, 1058, 32, 64] ,  [1, 1058, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1078, 32, 64] ,  [1, 1078, 32, 64] ,  [1, 1078, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1066, 32, 64] ,  [1, 1066, 32, 64] ,  [1, 1066, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1053, 32, 64] ,  [1, 1053, 32, 64] ,  [1, 1053, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1042, 32, 64] ,  [1, 1042, 32, 64] ,  [1, 1042, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1067, 32, 64] ,  [1, 1067, 32, 64] ,  [1, 1067, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1035, 32, 64] ,  [1, 1035, 32, 64] ,  [1, 1035, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1041, 32, 64] ,  [1, 1041, 32, 64] ,  [1, 1041, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1062, 32, 64] ,  [1, 1062, 32, 64] ,  [1, 1062, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1037, 32, 64] ,  [1, 1037, 32, 64] ,  [1, 1037, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1034, 32, 64] ,  [1, 1034, 32, 64] ,  [1, 1034, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1052, 32, 64] ,  [1, 1052, 32, 64] ,  [1, 1052, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1051, 32, 64] ,  [1, 1051, 32, 64] ,  [1, 1051, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1036, 32, 64] ,  [1, 1036, 32, 64] ,  [1, 1036, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1048, 32, 64] ,  [1, 1048, 32, 64] ,  [1, 1048, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1056, 32, 64] ,  [1, 1056, 32, 64] ,  [1, 1056, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1057, 32, 64] ,  [1, 1057, 32, 64] ,  [1, 1057, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1069, 32, 64] ,  [1, 1069, 32, 64] ,  [1, 1069, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1065, 32, 64] ,  [1, 1065, 32, 64] ,  [1, 1065, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1050, 32, 64] ,  [1, 1050, 32, 64] ,  [1, 1050, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1076, 32, 64] ,  [1, 1076, 32, 64] ,  [1, 1076, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1044, 32, 64] ,  [1, 1044, 32, 64] ,  [1, 1044, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1046, 32, 64] ,  [1, 1046, 32, 64] ,  [1, 1046, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1090, 32, 64] ,  [1, 1090, 32, 64] ,  [1, 1090, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1061, 32, 64] ,  [1, 1061, 32, 64] ,  [1, 1061, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1049, 32, 64] ,  [1, 1049, 32, 64] ,  [1, 1049, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1077, 32, 64] ,  [1, 1077, 32, 64] ,  [1, 1077, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1039, 32, 64] ,  [1, 1039, 32, 64] ,  [1, 1039, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1055, 32, 64] ,  [1, 1055, 32, 64] ,  [1, 1055, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2955, 32, 64] ,  [1, 2955, 32, 64] ,  [1, 2955, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1063, 32, 64] ,  [1, 1063, 32, 64] ,  [1, 1063, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1072, 32, 64] ,  [1, 1072, 32, 64] ,  [1, 1072, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2377, 32, 64] ,  [1, 2377, 32, 64] ,  [1, 2377, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1070, 32, 64] ,  [1, 1070, 32, 64] ,  [1, 1070, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1081, 32, 64] ,  [1, 1081, 32, 64] ,  [1, 1081, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2913, 32, 64] ,  [1, 2913, 32, 64] ,  [1, 2913, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2983, 32, 64] ,  [1, 2983, 32, 64] ,  [1, 2983, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1025, 5, 128] ,  [1, 1025, 5, 128] ,  [1, 1025, 5, 128] ,  0.0,  False,  False,  ), 
    (  [1, 3041, 32, 64] ,  [1, 3041, 32, 64] ,  [1, 3041, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2339, 32, 64] ,  [1, 2339, 32, 64] ,  [1, 2339, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2287, 32, 64] ,  [1, 2287, 32, 64] ,  [1, 2287, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 1060, 32, 64] ,  [1, 1060, 32, 64] ,  [1, 1060, 32, 64] ,  0.0,  True,  False,  ), 
    (  [1, 2745, 32, 64] ,  [1, 2745, 32, 64] ,  [1, 2745, 32, 64] ,  0.0,  True,  False,  ), 
]