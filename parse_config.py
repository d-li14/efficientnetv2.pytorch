
#################### EfficientNet V2 configs ####################
v2_base_block = [  # The baseline config for v2 models.
    'r1_k3_s1_e1_i32_o16_c1',
    'r2_k3_s2_e4_i16_o32_c1',
    'r2_k3_s2_e4_i32_o48_c1',
    'r3_k3_s2_e4_i48_o96_se0.25',
    'r5_k3_s1_e6_i96_o112_se0.25',
    'r8_k3_s2_e6_i112_o192_se0.25',
]


v2_s_block = [  # about base * (width1.4, depth1.8)
    'r2_k3_s1_e1_i24_o24_c1',
    'r4_k3_s2_e4_i24_o48_c1',
    'r4_k3_s2_e4_i48_o64_c1',
    'r6_k3_s2_e4_i64_o128_se0.25',
    'r9_k3_s1_e6_i128_o160_se0.25',
    'r15_k3_s2_e6_i160_o256_se0.25',
]


v2_m_block = [  # about base * (width1.6, depth2.2)
    'r3_k3_s1_e1_i24_o24_c1',
    'r5_k3_s2_e4_i24_o48_c1',
    'r5_k3_s2_e4_i48_o80_c1',
    'r7_k3_s2_e4_i80_o160_se0.25',
    'r14_k3_s1_e6_i160_o176_se0.25',
    'r18_k3_s2_e6_i176_o304_se0.25',
    'r5_k3_s1_e6_i304_o512_se0.25',
]


v2_l_block = [  # about base * (width2.0, depth3.1)
    'r4_k3_s1_e1_i32_o32_c1',
    'r7_k3_s2_e4_i32_o64_c1',
    'r7_k3_s2_e4_i64_o96_c1',
    'r10_k3_s2_e4_i96_o192_se0.25',
    'r19_k3_s1_e6_i192_o224_se0.25',
    'r25_k3_s2_e6_i224_o384_se0.25',
    'r7_k3_s1_e6_i384_o640_se0.25',
]

v2_xl_block = [  # only for 21k pretraining.
    'r4_k3_s1_e1_i32_o32_c1',
    'r8_k3_s2_e4_i32_o64_c1',
    'r8_k3_s2_e4_i64_o96_c1',
    'r16_k3_s2_e4_i96_o192_se0.25',
    'r24_k3_s1_e6_i192_o256_se0.25',
    'r32_k3_s2_e6_i256_o512_se0.25',
    'r8_k3_s1_e6_i512_o640_se0.25',
]

for blk in [v2_base_block, v2_s_block, v2_m_block, v2_l_block, v2_xl_block]:
    cfgs = []
    for k in blk:
        keys = k.split('_')
        cfgs.append([int(keys[3][1:]), int(keys[5][1:]), int(keys[0][1:]), int(keys[2][1:]), 1 if 'se' in keys[-1] else 0])
    print(cfgs)
