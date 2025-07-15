from . import test_samplers

def add_zamplers(NODE_CLASS_MAPPINGS, extra_samplers):
    
    NODE_CLASS_MAPPINGS.update({
        "SamplerRK_Test"                    : test_samplers.SamplerRK_Test,
        "Zampler_Test"                      : test_samplers.Zampler_Test,
        "Zampler"                           : test_samplers.Zampler,
        "UltraSharkSamplerRBTest"           : test_samplers.UltraSharkSamplerRBTest,
    })

    extra_samplers.update({
        "rk_test"                           : test_samplers.sample_rk_test,
        "rk_vptest3"                        : test_samplers.sample_rk_vptest3,
        "rk_unibutt"                        : test_samplers.sample_rk_unibutt,
        "rk_vptest"                         : test_samplers.sample_rk_vptest,
        "rk_vptest2"                        : test_samplers.sample_rk_vptest2,
        "rk_uniuni"                         : test_samplers.sample_rk_uniuni,
        "rk_sphere"                         : test_samplers.sample_rk_sphere,
        "rk_vpsde"                          : test_samplers.sample_rk_vpsde,
        "rk_vpsde_ddpm"                     : test_samplers.sample_rk_vpsde_ddpm,
        "rk_vpsde_csbw"                     : test_samplers.sample_rk_vpsde_csbw,
        "rk_momentum"                       : test_samplers.sample_rk_momentum,
        "rk_ralston_2s"                     : test_samplers.sample_rk_ralston_2s,
        "rk_implicit_res_2s"                : test_samplers.sample_rk_implicit_res_2s,
        "res_multistep"                     : test_samplers.sample_res_multistep,
        "rk_res_2m"                         : test_samplers.sample_rk_res_2m,
        "rk_res_2s"                         : test_samplers.sample_rk_res_2s,
        "rk_res_2s_prenoise"                : test_samplers.sample_rk_res_2s_prenoise,
        "rk_ralston_2s_prenoise"            : test_samplers.sample_rk_ralston_2s_prenoise,

        "rk_ralston_2s_prenoise_alt"        : test_samplers.sample_rk_ralston_2s_prenoise_alt,
        "rk_ralston_2s_prenoise_alt2"       : test_samplers.sample_rk_ralston_2s_prenoise_alt2,
        "rk_ralston_3s_prenoise_alt2"       : test_samplers.sample_rk_ralston_3s_prenoise_alt2,

        "rk_res_2m_nonstandard_prenoise_alt": test_samplers.sample_rk_res_2m_nonstandard_prenoise_alt,

        "rk_res_2m_prenoise_alt"            : test_samplers.sample_rk_res_2m_prenoise_alt,
        "rk_res_2m_prenoise_alt2"           : test_samplers.sample_rk_res_2m_prenoise_alt2,

        "rk_res_2s_prenoise_alt"            : test_samplers.sample_rk_res_2s_prenoise_alt,
        "rk_res_2s_prenoise_data"           : test_samplers.sample_rk_res_2s_prenoise_data,

        "rk_res_2s_scaled"                  : test_samplers.sample_rk_res_2s_scaled,
        "rk_res_2m_scaled"                  : test_samplers.sample_rk_res_2m_scaled,

        "rk_res_2s_overstep"                : test_samplers.sample_rk_res_2s_overstep,
        "rk_res_2s_downswap"                : test_samplers.sample_rk_res_2s_downswap,
        
        "rk_fedit_euler"                    : test_samplers.sample_rk_fedit_euler,

        "rk_fedit"                          : test_samplers.sample_rk_fedit,
        "rk_zample_edit2"                   : test_samplers.sample_zample_edit2,
        "rk_flow"                           : test_samplers.sample_rk_flow,
        "rk_flow2"                          : test_samplers.sample_rk_flow2,
        "rk_flow3"                          : test_samplers.sample_rk_flow3,

        "rk_triflow"                        : test_samplers.sample_rk_triflow,
        "rk_triflow_intersection"           : test_samplers.sample_rk_triflow_intersection,


        #"rk_flow_midpoint"                  : test_samplers.sample_rk_flow_midpoint,
        "rk_flow_ralston_2s"                : test_samplers.sample_rk_flow_ralston_2s,
        "rk_flow_ralston_2s_redo"           : test_samplers.sample_rk_flow_ralston_2s_redo,
        "rk_flow_ralston_3s_redo"           : test_samplers.sample_rk_flow_ralston_3s_redo,
        "rk_flow_ralston_4s_redo"           : test_samplers.sample_rk_flow_ralston_4s_redo,
        "rk_flow_gauss_2s"                  : test_samplers.sample_rk_flow_gauss_2s,


        "rk_flow_works"                     : test_samplers.sample_rk_flow_works,

        
        "rk_crazy"                          : test_samplers.sample_rk_crazy,
        "rk_crazy2"                         : test_samplers.sample_rk_crazy2,
        "rk_crazymod43"                     : test_samplers.sample_rk_crazymod43,
        "rk_crazymod44"                     : test_samplers.sample_rk_crazymod44,
        "rk_crazymod45"                     : test_samplers.sample_rk_crazymod45,
        "rk_pec423"                         : test_samplers.sample_rk_pec423,
        "rk_pec433"                         : test_samplers.sample_rk_pec433,
        "rk_gausslang_full"                 : test_samplers.sample_rk_gausslang_full,
        "rk_gausslang_3s_full"              : test_samplers.sample_rk_gausslang_3s_full,
        "rk_gausslang_3s_full_guide"        : test_samplers.sample_rk_gausslang_3s_full_guide,
        "rk_radau_iia_alt_lang_3s_full"     : test_samplers.sample_rk_radau_iia_alt_lang_3s_full,
        "rk_implicit"                       : test_samplers.sample_rk_implicit,

        "er_sde"                            : test_samplers.sample_er_sde,
        "er_sde_comfy"                      : test_samplers.sample_er_sde_comfy,

        "rk_gausslang"                      : test_samplers.sample_rk_gausslang,
        "rk_gausslangeps"                   : test_samplers.sample_rk_gausslangeps,
        "rk_ralradau"                       : test_samplers.sample_rk_ralradau,


        "rk_gausscycle"                     : test_samplers.sample_rk_gausscycle,
        "rk_gausscycle2"                    : test_samplers.sample_rk_gausscycle2,

        "rk_radaucycle"                     : test_samplers.sample_rk_radaucycle,
        "rk_radaucycle_ia"                  : test_samplers.sample_rk_radaucycle_ia,
        "rk_radaucycle_3s"                  : test_samplers.sample_rk_radaucycle_3s,
        "rk_radaucycle_retry"               : test_samplers.sample_rk_radaucycle_retry,
        "rk_radaucycle_staggered"           : test_samplers.sample_rk_radaucycle_staggered,
        
        "rk_implicit_euler_von_svd"         : test_samplers.sample_rk_implicit_euler_von_svd,
        "rk_implicit_cycloeuler"            : test_samplers.sample_rk_implicit_cycloeuler,
        "rk_salmon"                         : test_samplers.sample_rk_salmon,


        "rk_radau_iia_2s_BS"                : test_samplers.sample_rk_radau_iia_2s_BS,

        "rk_radau_ia_2s_lang_full"          : test_samplers.sample_rk_radau_ia_2s_lang_full,

        "rk_radau_iia_2s_lang_full"         : test_samplers.sample_rk_radau_iia_2s_lang_full,

        "rk_radau_iia_2s"                   : test_samplers.sample_rk_radau_iia_2s,
        "rk_radau_iia_3s"                   : test_samplers.sample_rk_radau_iia_3s,


        "rk_abnorsett4"                     : test_samplers.sample_rk_abnorsett4,
        "rk_euler_lowranksvd"               : test_samplers.sample_rk_euler_lowranksvd,
        "rk_randomized_svd"                 : test_samplers.sample_rk_randomized_svd,
        "rk_implicit_euler_fd"              : test_samplers.sample_rk_implicit_euler_fd,

        "rk_euler_banana_alphaupdown_test"  : test_samplers.sample_rk_euler_banana_alphaupdown_test,
        "rk_euler_hamberder"                : test_samplers.sample_rk_euler_hamberder,

        "rk_euler_banana"                   : test_samplers.sample_rk_euler_banana,
        "rk_res_2s_banana"                  : test_samplers.sample_rk_res_2s_banana,
        "rk_res_3s_banana"                  : test_samplers.sample_rk_res_3s_banana,

        "rk_euler"                          : test_samplers.sample_rk_euler,

        "rk_euler_prenoise"                 : test_samplers.sample_rk_euler_prenoise,


        "rk_ddim_test"                      : test_samplers.sample_rk_ddim_test,
        "rk_res_denoise_eps"                : test_samplers.sample_rk_res_denoise_eps,
        "rk_exp_euler_denoise_eps"          : test_samplers.sample_rk_exp_euler_denoise_eps,
        "rk_euler_alt_sde"                  : test_samplers.sample_rk_euler_alt_sde,

        "rk_implicit_euler"                 : test_samplers.sample_rk_implicit_euler,

        "rk_vpsde_trivial"                  : test_samplers.sample_rk_vpsde_trivial,
        "zample"                            : test_samplers.sample_zsample,
        "zample_paper"                      : test_samplers.sample_zample_paper,
        "zample_inversion"                  : test_samplers.sample_zample_inversion,
        "sample_zample_edit"                : test_samplers.sample_zample_edit,
        
    })
    
    return NODE_CLASS_MAPPINGS, extra_samplers

