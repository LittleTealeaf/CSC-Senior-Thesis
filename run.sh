echo "epoch,time,variables,bootstraps" > "$OUT_PATH/times.csv"
for vars in $(head -n 1 "$DATA_PATH/index" | tr ',' '\n'); do
    for bootstraps in $(tail -n 1 "$DATA_PATH/index" | tr ',' '\n'); do
				echo "Executing $NAME for $vars Variables and $bootstraps Bootstraps"
        OUT_PATH="$OUT_PATH/tmp" DATASET_PATH="$DATA_PATH/vars/$vars/data" NETWORK_PATH="$DATA_PATH/vars/$vars/network" BOOTSTRAP_COUNT="$bootstraps" BOOTSTRAP_PATH="$DATA_PATH/bootstraps" bash $SCRIPT
        awk "{print \$0 \",$vars,$bootstraps\"}" "$OUT_PATH/tmp" >> "$OUT_PATH/times.csv"
    done
done
rm "$OUT_PATH/tmp"
