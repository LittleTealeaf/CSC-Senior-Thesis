rm $OUT_FILE 2> /dev/null || true
for vars in $(head -n 1 "$DATA_PATH/index" | tr ',' '\n'); do
    for bootstraps in $(tail -n 1 "$DATA_PATH/index" | tr ',' '\n'); do
				echo "Executing $NAME for $vars Variables and $bootstraps Bootstraps"
				TEMP_PATH=$(mktemp)

				BOOTSTRAP_PATH="$DATA_PATH/bootstraps" DATASET_PATH="$DATA_PATH/vars/$vars/data" NETWORK_PATH="$DATA_PATH/vars/$vars/network" BOOTSTRAP_COUNT="$bootstraps" OUT_PATH="$TEMP_PATH" bash $SCRIPT

				awk "{print \$0 \",$vars,$bootstraps,$NAME\"}" "$TEMP_PATH" >> $OUT_FILE




        # OUT_PATH="$OUT_PATH/tmp" DATASET_PATH="$DATA_PATH/vars/$vars/data" NETWORK_PATH="$DATA_PATH/vars/$vars/network" BOOTSTRAP_COUNT="$bootstraps" BOOTSTRAP_PATH="$DATA_PATH/bootstraps" bash $SCRIPT
        # awk "{print \$0 \",$vars,$bootstraps\"}" "$OUT_PATH/tmp" >> "$OUT_PATH/times.csv"
    done
done
