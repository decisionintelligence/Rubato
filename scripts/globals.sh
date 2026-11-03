# define some globally shared, fixed information for all scripts to lookup
get_dataset_info() {
    local dataset_name=$1
    local dataset_subset_name=$2
    
    case $dataset_name in
        P12)
            dataset_root_path="storage/datasets/P12"
            n_variables=41
            ;;
        MIMIC)
            dataset_root_path="storage/datasets/MIMIC"
            n_variables=96
            ;;
        ECL)
            dataset_root_path="storage/datasets/electricity"
            n_variables=321
            ;;
        ETTh1|ETTm1)
            dataset_root_path="storage/datasets/ETT-small"
            n_variables=7
            ;;
        HumanActivity)
            dataset_root_path="storage/datasets/HumanActivity"
            n_variables=12
            ;;
        ILI)
            dataset_root_path="storage/datasets/illness"
            n_variables=7
            ;;
        MIMIC_III)
            dataset_root_path="storage/datasets/MIMIC_III"
            n_variables=96
            ;;
        MIMIC_IV)
            dataset_root_path="storage/datasets/MIMIC_IV"
            n_variables=100
            ;;
        Traffic)
            dataset_root_path="storage/datasets/traffic"
            n_variables=862
            ;;
        USHCN)
            dataset_root_path="storage/datasets/USHCN"
            n_variables=5
            ;;
        Weather)
            dataset_root_path="storage/datasets/weather"
            n_variables=21
            ;;
        CESNET)
            dataset_root_path="storage/datasets/CESNET"
            n_variables=5
            ;;
        EPA_AIR)
            dataset_root_path="storage/datasets/EPA_AIR"
            n_variables=4
            ;;
        FNSPID)
            dataset_root_path="storage/datasets/FNSPID"
            n_variables=6
            ;;
        PAMP2)
            dataset_root_path="storage/datasets/PAMP2"
            n_variables=4
            ;;
        ClusterTrace)
            dataset_root_path="storage/datasets/ClusterTrace"
            n_variables=3
            ;;
        GDELT)
            dataset_root_path="storage/datasets/GDELT"
            n_variables=4
            ;;
        *)
            echo "Error: Dataset '$dataset_name' not found in scripts/globals.sh" >&2
            return 1
            ;;
    esac
    
    return 0
}