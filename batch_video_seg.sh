#!/bin/bash

# Function to submit a job using sbatch
submit_job() {
    local object_type=$1
    local data_dir=$2
    local x_coords=$3
    local y_coords=$4
    
    # Create a temporary script file
    TMP_SCRIPT="/svl/u/tarunc/tool_use_benchmark/sam2/sam_seg_${object_type}_$(basename ${data_dir})_$(date +%s).sh"
    
    # Write the sbatch script
    cat > $TMP_SCRIPT << EOL
#!/bin/bash
#SBATCH --account viscam 
#SBATCH --job-name sam_${object_type}
#SBATCH --partition=svl 
#SBATCH --gres=gpu:1 
#SBATCH --mem=64G
#SBATCH --exclude=svl17,svl3,svl5,svl6,svl4
#SBATCH --output=/svl/u/tarunc/tool_use_benchmark/FoundationPose/slurm_outs/%j.out
#SBATCH --error=/svl/u/tarunc/tool_use_benchmark/FoundationPose/slurm_outs/%j.err

cd /svl/u/tarunc/tool_use_benchmark/sam2
source ~/.bashrc 
conda activate sam2

python video_segmentation.py \\
    --out_dir=${data_dir} \\
    -x '${x_coords}' \\
    -y '${y_coords}' \\
    --t=${object_type}
EOL

    # Submit the job
    echo "Submitting ${object_type} segmentation job for $(basename ${data_dir})"
    sbatch $TMP_SCRIPT
    
    # Clean up the temporary script
    rm $TMP_SCRIPT
}


# Directory: d347da5c_bluescooper_bowl_nuts
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/d347da5c_bluescooper_bowl_nuts"
# submit_job "tool" "$DATA_DIR" "[[152, 176], [155, 145], [262, 323], [159], [228], [162, 166], [136, 129, 159], [494]]" "[[218, 200], [252, 238], [361, 354], [358], [129], [253, 236], [309, 317, 299], [224]]"
# submit_job "target" "$DATA_DIR" "[[288], [346], [261], [271], [369], [353], [302], [446]]" "[[223], [233], [148], [167], [185], [247], [308], [299]]"

# Directory: dfa924b3_scoop_sand2
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/dfa924b3_scoop_sand2"
# submit_job "tool" "$DATA_DIR" "[[161], [164], [369], [179], [244], [166], [150], [512]]" "[[217], [255], [339], [365], [131], [254], [308], [233]]"
# submit_job "target" "$DATA_DIR" "[[293], [333], [212], [278], [363], [350], [302], [420]]" "[[219], [233], [188], [192], [219], [240], [283], [361]]"

# Directory: cfbef9d1_scoop_coffee5
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/cfbef9d1_scoop_coffee5"
# submit_job "tool" "$DATA_DIR" "[[127], [174], [408], [243], [205], [162], [128], [452]]" "[[233], [280], [307], [383], [166], [272], [328], [227]]"
# submit_job "target" "$DATA_DIR" "[[297], [345], [225], [257], [369], [354], [309], [423]]" "[[224], [220], [140], [148], [169], [229], [276], [304]]"

# Directory: b0667b82_scoop_icecream1
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/b0667b82_scoop_icecream1"
# submit_job "tool" "$DATA_DIR" "[[142], [159], [390], [202], [230], [157], [132], [481]]" "[[223], [275], [340], [383], [138], [267], [322], [226]]"
# submit_job "target" "$DATA_DIR" "[[276], [341], [251], [261], [370], [354], [307], [430]]" "[[205], [219], [138], [156], [174], [228], [271], [298]]"

# Directory: b26ae22a_scoop_sand4
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/b26ae22a_scoop_sand4"
# submit_job "tool" "$DATA_DIR" "[[150], [154], [368], [171], [234], [159], [143], [505]]" "[[213], [244], [348], [361], [117], [245], [308], [230]]"
# submit_job "target" "$DATA_DIR" "[[305], [337], [218], [263], [379], [351], [303], [427]]" "[[201], [233], [152], [154], [169], [235], [284], [297]]"

# Directory: 72031343_scoop_icecream6
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/72031343_scoop_icecream6"
# submit_job "tool" "$DATA_DIR" "[[154], [155], [378], [185], [230], [168], [147], [501]]" "[[217], [261], [334], [371], [126], [254], [312], [229]]"
# submit_job "target" "$DATA_DIR" "[[315], [358], [228], [280], [403], [391], [330], [409]]" "[[208], [210], [130], [152], [218], [237], [274], [306]]"

# Directory: 242181cc_scoop_icecream4
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/242181cc_scoop_icecream4"
# submit_job "tool" "$DATA_DIR" "[[166], [161], [384], [183], [231], [164], [142], [492]]" "[[208], [248], [330], [357], [126], [248], [308], [223]]"
# submit_job "target" "$DATA_DIR" "[[306], [364], [234], [279], [369], [345], [319], [417]]" "[[202], [211], [137], [145], [169], [225], [270], [289]]"

# Directory: 34606da3_scoop_sand5
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/34606da3_scoop_sand5"
# submit_job "tool" "$DATA_DIR" "[[154], [157], [377], [193], [234], [167], [138], [500]]" "[[222], [268], [343], [372], [132], [265], [231], [236]]"
# submit_job "target" "$DATA_DIR" "[[296], [340], [224], [269], [378], [364], [314], [423]]" "[[206], [216], [133], [152], [167], [231], [274], [300]]"

# Directory: 16713eef_scoop_coffee3
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/16713eef_scoop_coffee3"
# submit_job "tool" "$DATA_DIR" "[[158], [164], [388], [181], [234], [172], [144], [504]]" "[[216], [263], [325], [366], [123], [250], [307], [224]]"
# submit_job "target" "$DATA_DIR" "[[290], [342], [229], [255], [375], [361], [317], [439]]" "[[208], [212], [124], [151], [165], [238], [278], [298]]"

# Directory: 30d7c35a_scoop_coffee1
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/30d7c35a_scoop_coffee1"
# submit_job "tool" "$DATA_DIR" "[[135], [152], [410], [211], [212], [151], [119], [483]]" "[[224], [272], [348], [396], [136], [269], [324], [223]]"
# submit_job "target" "$DATA_DIR" "[[307], [359], [216], [263], [385], [377], [305], [422]]" "[[201], [215], [134], [148], [172], [229], [271], [310]]"

# Directory: 9e8d2e67_scoop_sand6
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/9e8d2e67_scoop_sand6"
# submit_job "tool" "$DATA_DIR" "[[129], [162], [424], [203], [210], [158], [122], [473]]" "[[226], [278], [326], [383], [152], [277], [331], [228]]"
# submit_job "target" "$DATA_DIR" "[[229], [356], [212], [268], [374], [360], [313], [422]]" "[[201], [213], [148], [147], [167], [233], [276], [302]]"

# Directory: 1aac3dc9_scoop_icecream3
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/1aac3dc9_scoop_icecream3"
# submit_job "tool" "$DATA_DIR" "[[160], [157], [396], [196], [227], [167], [153], [491]]" "[[204], [256], [333], [386], [129], [235], [296], [224]]"
# submit_job "target" "$DATA_DIR" "[[284], [345], [233], [271], [353], [362], [302], [422]]" "[[215], [219], [130], [149], [191], [234], [277], [300]]"

# Directory: 0fdae376_blackcup_plate_coffee
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/0fdae376_blackcup_plate_coffee"
# submit_job "tool" "$DATA_DIR" "[[179], [216], [297], [219], [255], [233], [177], [481]]" "[[172], [232], [288], [288], [138], [241], [293], [245]]"
# submit_job "target" "$DATA_DIR" "[[278], [347], [268], [277], [359], [358], [302], [412]]" "[[243], [256], [186], [198], [228], [267], [315], [342]]"

# Directory: 25d59628_brush_pan_clay
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/25d59628_brush_pan_clay"
# submit_job "tool" "$DATA_DIR" "[[193, 171, 204], [192, 209], [252], [175, 186], [279], [208, 200, 225], [185, 164], [513]]" "[[213, 216, 220], [247, 243], [323], [327, 307], [137], [244, 242, 250], [307, 312], [260]]"
# submit_job "target" "$DATA_DIR" "[[361], [391], [211], [285], [443], [421], [390], [418]]" "[[236], [231], [126], [144], [236], [258], [295], [430]]"

# Directory: b9c595b1_knife_plate_clay
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/b9c595b1_knife_plate_clay"
# submit_job "tool" "$DATA_DIR" "[[197], [205, 190, 210], [270], [199], [265], [213], [182], [499]]" "[[215], [254, 239, 261], [311], [324], [150], [261], [310], [264]]"
# submit_job "target" "$DATA_DIR" "[[226], [284], [253], [256], [297], [291], [304], [440]]" "[[247], [248], [204], [231], [232], [270], [310], [340]]"

# Directory: 1202ba72_shotglass_plate_coffee
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/1202ba72_shotglass_plate_coffee"
# submit_job "tool" "$DATA_DIR" "[[188], [230], [284], [233], [270], [233], [189], [466]]" "[[217], [257], [282], [309], [158], [259], [300], [257]]"
# submit_job "target" "$DATA_DIR" "[[279], [337], [271], [289], [370], [362], [316], [405]]" "[[243], [247], [184], [207], [219], [265], [312], [359]]"

# Directory: a8dfe5f4_grayduster
# DATA_DIR="/svl/u/tarunc/tool_use_benchmark/FoundationPose/demo_data/new_data/a8dfe5f4_grayduster"
# submit_job "tool" "$DATA_DIR" "[[141], [219], [326], [218], [215], [195], [128], [453]]" "[[217], [273], [309], [359], [153], [255], [304], [218]]"