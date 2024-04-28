import os
import glob
import csv
from pathlib import Path
import dateparser
import time
from datetime import datetime
from StyleFrame import StyleFrame
import pandas as pd
import mxnet as mx
import numpy as np
import re
import yaml
from collections import OrderedDict
from daterangeparser import parse_date_range
from ai_document_parser.models.predict_model import PredictDocument
from ai_document_parser.models.predict_kv_invoice import run_key_value_inference, get_fields_from_crops
# from ai_document_classifier.models.predict_model import DocumentTypeClassifier
from ai_document_parser.models.predict_kv_invoice import parse_amount
from ai_text_classifier.models import predict_model as ai_text_classifier_inference
from ai_text_semantic_extractor.models import predict_model as ai_text_semantic_extractor_inference
from ai_text_classifier.models.predict_model import PredictColumn
from paperentry_backend.utils import get_update_code
from dc_dataset_utils import moneyparser
from PIL import Image
from dcutils.font import resize_contain

IRS_OUTPUT_COLUMNS = [
    "Account Bank Name", "Account Holder Name", "Account Number", "Account Statement Period Start",
    "Account Statement Period End", "Debit", "Credit", "Deposits", "Balance", "Statement Starting Balance",
    "Statement Ending Balance", "Transaction Date", "Transaction Type", "Transaction Vendor Name", "Transaction Note",
    "Check Number", "Check Payee Name", "Check Payee Address", "Check Amount", "Image", "Filename"
]
ai_ctx = mx.cpu()
GPU = os.environ.get("GPU", None)
if GPU and GPU != "-1":
    # os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU")
    ai_ctx = mx.gpu(int(GPU))
HOCR_DFS = {}
OCR_THRESH = 50

COLUMN_CLASSIFIER_WEIGHT_FILE = "table_column_classifier_11-20-2019-TIME-11AM_50_01_keras_text_classifier-EPOCH_94-VAL_ACC_0.99.hdf5"
COLUMN_CLASSIFIER_LABELS = "table_column_classifier_11-20-2019-TIME-11AM_50_01_labels.pickle"
COLUMN_CLASSIFIER_TOKENIZER = "table_column_classifier_11-20-2019-TIME-11AM_50_01_tokenizer.pickle"

BANKSTATEMENT_TABLE_PARAMS = "AI_IRS_BANKSTATEMENT_TABLE_DATASET_11-12-2019-TIME-11PM_59_48mask_rcnn_fpn_resnet101_v1d_coco_best.params"
BANKSTATEMENT_TABLE_MAPPING = "AI_IRS_BANKSTATEMENT_TABLE_DATASET_11-12-2019-TIME-11PM_59_48_class_id_to_name_mapping.yaml"
BANKSTATEMENT_KV_PARAMS = \
    "AI_IRS_BANKSTATEMENT_KV_REQUIRED_ONLY_DATASET_11-26-2019-TIME-01PM_27_58mask_rcnn_fpn_resnet101_v1d_coco_0018_18.1000.params"
    #"AI_IRS_BANKSTATEMENT_KV_REQUIRED_ONLY_DATASET_11-21-2019-TIME-03AM_02_16mask_rcnn_fpn_resnet101_v1d_coco_best.params"
BANKSTATEMENT_KV_MAPPING = \
    "AI_IRS_BANKSTATEMENT_KV_REQUIRED_ONLY_DATASET_11-26-2019-TIME-01PM_27_58_class_id_to_name_mapping.yaml"
    #"AI_IRS_BANKSTATEMENT_KV_REQUIRED_ONLY_DATASET_11-21-2019-TIME-03AM_02_16_class_id_to_name_mapping.yaml"

DOCUMNET_CLASSIFIER_PARAMS = "/data/paperentry_irs/weights/document_type_0095_best.params-0095.params"
DOCUMNET_CLASSIFIER_JSON = "/data/paperentry_irs/weights/document_type_0095_best.params-symbol.json"
DOCUMNET_CLASSIFIER_LABELS = "/data/paperentry_irs/weights/document_type_labels.pickle"

KV_PARSER_PARAMS = 'KV_NO_TBL_ADDR_mask_rcnn_fpn_resnet101_v1d_coco_best.params'
KV_EXTRACTOR_CONFIG_YAML = 'KV_KEYS_VALUES_generator_config.yaml'


def save_img_crops(original_img, basename, row_index, class_name, crop_bbox, out_dir):

    crops = []

    crop_img = original_img.crop(tuple(crop_bbox))
    w, h = crop_img.size
    if w <= 0 or h <= 0:
        print("Error in Cropping: " + basename + str(row_index) + "_" + class_name)
        return ""
    crop_img = resize_contain(crop_img, size=(1024, 256))
    crop_image_name = os.path.join(out_dir,
                                   "{}_{}.png".format(row_index, class_name.replace("MAPPED_", ""),
                                                         ))
    crop_img.save(crop_image_name)
    return crop_image_name


def process_document(input_dir, output_dir, doc_type, weight_dir):
    get_update_code()

    bs_table_params_file = os.path.join(weight_dir, BANKSTATEMENT_TABLE_PARAMS)
    bs_table_class_mapping_file = os.path.join(weight_dir, BANKSTATEMENT_TABLE_MAPPING)
    bs_kv_params_file = os.path.join(weight_dir, BANKSTATEMENT_KV_PARAMS)
    bs_kv_class_mapping_file = os.path.join(weight_dir, BANKSTATEMENT_KV_MAPPING)

    document_params_file = os.path.join(weight_dir, DOCUMNET_CLASSIFIER_PARAMS)
    document_classifier_json = os.path.join(weight_dir, DOCUMNET_CLASSIFIER_JSON)
    document_type_labels_file = os.path.join(weight_dir, DOCUMNET_CLASSIFIER_LABELS)

    kv_params_file = os.path.join(weight_dir, KV_PARSER_PARAMS)
    kv_generator_config_file = os.path.join(weight_dir, KV_EXTRACTOR_CONFIG_YAML)

    if os.path.isdir(input_dir):
        _files = glob.glob(input_dir + "/*.[pP][dD][fF]")
    elif os.path.isfile(input_dir):
        _files = [input_dir]
    else:
        raise Exception("Invalid Input file/directory!")
    os.makedirs(output_dir, exist_ok=True)
    # if len(_files) > 1:
    #     print("Error: Only single pdf is supported")
    #     exit(-1)

    predDoc = PredictDocument(bs_table_params_file, bs_table_class_mapping_file, debug_mode=True)

    results_dfs = []
    TIME_STR = datetime.now().strftime("%m-%d-%Y-TIME-%I%p_%M_%S")
    temp_file_path = os.path.join(output_dir, "result_{}.xlsx".format(TIME_STR))
    out_file_path = os.path.join(output_dir, "result.xlsx")
    for input_file in _files:
        result_dict = {}
        basename = os.path.basename(input_file)
        files, hocr_dfs, crops_path, ai_output_dir = predDoc.preprocess_doc(input_file, output_dir=output_dir)
        encoded_file = ""
        if len(files) == 1:
            encoded_file = files[0]
        else:
            # take the first file only if multiple
            for f in files:
                if str(f).endswith('_p01.enc.png') or str(f).endswith('_p01.png'):
                    encoded_file = f
                    break

        # docuemntTypeClassifier = DocumentTypeClassifier(document_params_file, document_classifier_json,
        #                                                 document_type_labels_file)
        # input_file_name, document_type_pred, score = docuemntTypeClassifier.run_inference(encoded_file)
        # if document_type_pred == "bankstatements":
        #     document_type_predicted = "bank statements"
        # elif document_type_pred == "invoices":
        #     document_type_predicted = "bill"
        # elif document_type_pred == "receipts":
        #     document_type_predicted == "receipt"
        document_type_predicted = doc_type.lower()
        print("document_type_predicted", document_type_predicted)
        if document_type_predicted == "bank statements":

            kv_bs_predDoc = PredictDocument(bs_kv_params_file, bs_kv_class_mapping_file, debug_mode=True)

            crops_dict, inference_kv_df, inference_table_df_dict = predDoc.run_inference(files, crops_path, hocr_dfs,
                                                                                         ai_output_dir, min_score=0.6,
                                                                                         save_label="table")  # , input_file, output_dir= output_dir)
            _, inference_kv_df, _ = kv_bs_predDoc.run_inference(files, crops_path, hocr_dfs,
                                                                ai_output_dir, min_score=0.6, save_label="kv")
            result_df = process_bs_document(inference_kv_df, inference_table_df_dict, basename, ai_output_dir, weight_dir)
            # result_csv = os.path.join(ai_output_dir, basename + ".csv")
            # result_df.to_csv(result_csv, index=False)
            # generate_xlsx(result_csv, doc_type, out_file_path)
            results_dfs.append(result_df)
            # generate_xlsx_from_dfs([result_df], os.path.join(output_dir, basename + ".xlsx"))

        # elif document_type_predicted == 'bill' or document_type_predicted == 'receipt':
        else:
            config = yaml.load(open(kv_generator_config_file, 'r'), Loader=yaml.Loader)
            class_names = OrderedDict([(cname, i) for i, cname in enumerate(config['FIELD_MAPPING'])])
            filter_class_ids = [class_names['KV'], class_names['KEY'], class_names['VALUE'], class_names['ADDR_VND'],
                                class_names['ADDRESSES'], class_names['ANK_KV'],
                                class_names["ANK_KEY"], class_names["ANK_VALUE"]]
            crops_path = os.path.join(ai_output_dir, 'crops')
            preprocess_out_dir = os.path.join(ai_output_dir, 'preprocessed')
            os.makedirs(crops_path, exist_ok=True)
            files = glob.glob(preprocess_out_dir + "/*.enc.png")
            print("Running predict for " + str(len(files)) + " files")

            crops_dict, text_dfs = run_key_value_inference(kv_params_file, files, filter_class_ids, crops_path,
                                                           class_names)
            out_file_path = process_kv_document(crops_dict, basename, output_dir, ai_output_dir,
                                                document_type_predicted, weight_dir, out_file_path)

        # else:
        #     pass
    if document_type_predicted == "bank statements":
        temp_file_path = generate_xlsx_from_dfs(results_dfs, temp_file_path)
        out_file_path = combine_excel(out_file_path, temp_file_path)
        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(e)

    return out_file_path


def process_kv_document(crops_dict, basename, output_dir, ai_output_dir, document_type_predicted, weight_dir,
                        out_file_path):
    crop_text_csv_file = os.path.join(ai_output_dir, "crops", "crop_text.csv")
    text_dict, vendor_info = post_prcess_crop_text_csv_old(crop_text_csv_file)
    print("text_dict", text_dict, "vendor_info", vendor_info)
    classifier_path = ai_output_dir + "/classes"
    os.makedirs(classifier_path, exist_ok=True)

    vendor_info_text = str(" ".join(vendor_info)).lower().strip()
    vendor_weight_file_path = os.path.join(weight_dir, "vendor_name_weight.hdf5")
    vendor_tokenizer_file_path = os.path.join(weight_dir, "vendor_tokenizer.pickle")
    predictVendorText = ai_text_semantic_extractor_inference.PredictText(vendor_weight_file_path,
                                                                         vendor_tokenizer_file_path)
    vendor_result = predictVendorText.predict(*[vendor_info_text])

    vendor_name = vendor_result[0].get("prediction")

    fields_dict, crops_dict = run_text_classifier_inference_old(text_dict, crops_dict, classifier_path,
                                                                document_type_predicted, weight_dir)

    fields_dict['VENDOR'] = vendor_name
    print("fields_dict", fields_dict)

    csv_path = os.path.join(output_dir, basename + '.csv')
    if fields_dict:
        classes = {"KV_TA": "Bill Total", "KV_INVN": "Bill No.", "KV_INVD": "Bill Date", "KV_DUED": "Due Date",
                   "KV_PON": "PO#", "VENDOR": "Vendor Name"}
        final_dict = {}
        for k, v in fields_dict.items():
            if k in classes:
                final_dict[classes[k]] = v
            else:
                final_dict[k] = v
        if "KV_TA" in fields_dict:
            final_dict["Amount Due"] = fields_dict["KV_TA"]
        with open(csv_path, 'w') as f:
            w = csv.DictWriter(f, final_dict.keys())
            w.writeheader()
            w.writerow(final_dict)
        generate_xlsx(csv_path, document_type_predicted, out_file_path)
    return out_file_path


def post_process_repeating_columns(df, column_mapping):
    num_date = 0
    total_columns = len(column_mapping)
    out_col_mapping = {}
    out_df = None
    for col_index, col_type in column_mapping.items():
        if col_type == "ATOMIC_TBL_COLUMN_DATE":
            num_date += 1
    if num_date > 1:
        if float(total_columns) / float(num_date) < 2:
            out_df = df
            out_col_mapping = column_mapping
            print("Multiple date columns found but not enough repeatation of other columns ")
        elif total_columns % num_date != 0:
            print("Multiple date columns found but other columns are not repeating with same frequency")
            out_col_mapping = column_mapping
            out_df = df
        else:
            num_new_cols = int(total_columns / num_date)
            new_col_names = df.columns[:num_new_cols]

            new_rows_dict = {}  # []
            for row_index, row in df.iterrows():
                for table_repeat_index in range(num_date):
                    start_index = table_repeat_index * num_new_cols
                    new_row = row[start_index:(start_index + num_new_cols)]
                    if table_repeat_index not in new_rows_dict:
                        new_rows_dict[table_repeat_index] = [new_row]
                    else:
                        new_rows_dict[table_repeat_index].append(new_row)
            new_rows = []
            for k, v in new_rows_dict.items():
                new_rows += v
            out_df = pd.DataFrame(new_rows, columns=new_col_names)

            for j in range(num_new_cols):
                out_col_mapping[j] = column_mapping[j]
    else:
        out_df = df
        out_col_mapping = column_mapping
    return out_df, out_col_mapping


def parse_date(date_str):
    out_str = ""
    try:
        date_obj = dateparser.parse(date_str)
        if date_obj is None:
            out_str = date_str
        else:
            out_str = date_obj.strftime("%m/%d/%Y")
    except Exception as e:
        print("Date parsing failed", e)
        out_str = date_str

    return out_str


def parse_amount(amount_str):
    out_val = 0.0
    try:
        out_val = moneyparser.price_dec(amount_str)
        out_val = float(out_val)
    except Exception as e:
        print(e)
        out_val = amount_str
    return out_val


def process_bs_document(inference_kv_df, inference_table_df_dict, basename, output_dir, weight_dir):
    final_output = []
    bank_name = ""
    account_holder_name = ""
    account_number = ""
    account_type = ""
    st_period = ""
    st_date = ""
    st_end_date = ""
    st_start_date = ""
    debit = ""
    credit = ""
    starting_balance = ""
    ending_balance = ""
    st_date = ""

    for i, row in inference_kv_df.iterrows():
        if row.loc["BLOCK_NAME"] == "MAPPED_BLOCK_ENTITY_INFO_BANK":
            if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ENTITY_NAME":
                bank_name = row.loc["VALUE_GENERAL"]
        if row.loc["BLOCK_NAME"] == "MAPPED_BLOCK_ENTITY_INFO_ACCOUNT_HOLDER":
            if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ENTITY_NAME":
                account_holder_name = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ACC_NUM":
            account_number = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ACC_TYPE":
            account_type = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ST_PERIOD":
            st_period = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ST_START_DATE":
            st_start_date = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ST_DATE":
            st_date = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ST_END_DATE":
            st_end_date = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_STARTING_BALANCE":
            starting_balance = row.loc["VALUE_GENERAL"]
        if row.loc["GROUP_NAME"] == "MAPPED_GROUP_ENDING_BALANCE":
            ending_balance = row.loc["VALUE_GENERAL"]
    # result_dict["AI_1_RESULT"] = inference_result

    if st_period and (not st_end_date or not st_start_date):
        try:
            st_start_date, st_end_date = parse_date_range.parse(st_period)
            if not st_date:
                st_date = st_end_date
        except Exception as e:
            print(st_period, "date Split failed", e)
            if "/" in st_period:
                st_start_date, st_end_date = st_period.strip().split("-")
    if not st_end_date:
        st_end_date = parse_date(st_date)
    if st_end_date:
        st_end_date = parse_date(st_end_date)
    if st_start_date:
        st_start_date = parse_date(st_start_date)

    if st_date:
        st_date = parse_date(st_date)
    final_output.append({'Account Bank Name': bank_name, "Account Holder Name": account_holder_name,
                         "Account Number": account_number, "Account Statement Period Start": st_start_date,
                         "Account Statement Period End": st_end_date,
                         "Statement Starting Balance": starting_balance, "Statement Ending Balance": ending_balance,
                         "Balance": ending_balance,
                         "Transaction Date": st_date})

    MODEL_FILE_TO_LOAD = os.path.join(weight_dir, COLUMN_CLASSIFIER_WEIGHT_FILE)
    TOKENIZER_FILE = os.path.join(weight_dir, COLUMN_CLASSIFIER_TOKENIZER)
    UNIQUE_LABEL_FILE = os.path.join(weight_dir, COLUMN_CLASSIFIER_LABELS)
    pred_text = PredictColumn(MODEL_FILE_TO_LOAD, TOKENIZER_FILE, UNIQUE_LABEL_FILE)
    for enc_png, table_dfs_data in inference_table_df_dict.items():
        basename = enc_png[:-8]+".png"
        for table_index, table_df_data in enumerate(table_dfs_data):
            # print(enc_png, table_df_data)
            table_df = table_df_data.get("df")
            table_df.fillna('', inplace=True)
            table_class_name = table_df_data.get("class_name")
            table_spur = table_df_data.get("table_spur")
            has_header = table_df_data.get("has_header")
            rows_boxes = table_df_data.get("rows_boxes")

            if table_class_name in ["MAPPED_BLOCK_TBL_TABLE_ACCOUNT_SUMMARY"]:
                for row_index, row in table_df.iterrows():
                    row_len = len(row)

                    if row_len >= 3:
                        starting_balance = row.iloc[1]

            if table_class_name in ["MAPPED_BLOCK_TBL_TABLE_TRANSACTION", "MAPPED_BLOCK_TBL_TABLE_TRANSACTION_DEBIT",
                                    "MAPPED_BLOCK_TBL_TABLE_TRANSACTION_CREDIT"]:
                original_img_file = os.path.join(output_dir, "preprocessed", basename)
                original_img = Image.open(original_img_file)
                crop_dir = os.path.join(output_dir, "crops")
                os.makedirs(crop_dir, exist_ok=True)
                col_mapping = {}
                print(table_class_name, "\n", table_df)
                for index, col in enumerate(table_df.columns):
                    col_values = [col] + table_df.iloc[:, index].values.tolist()
                    print("col_values", col_values)
                    try:
                        c = pred_text.run_inference(*col_values)
                        print("predicted", c)
                        if c[1] > 0.6:
                            col_mapping[index] = c[0]
                        else:
                            col_mapping[index] = "ATOMIC_TBL_COLUMN"

                    except Exception as e:
                        print("column prediction failed", e)
                        col_mapping[index] = "ATOMIC_TBL_COLUMN"
                table_df, col_mapping = post_process_repeating_columns(table_df, col_mapping)

                col_mapping_list = list(dict(sorted(col_mapping.items())).values())
                column_header = table_df.columns
                if not has_header:
                    table_df = pd.DataFrame(
                        np.row_stack([column_header, table_df.values]),
                        columns=col_mapping_list
                    )
                for row_index, row in table_df.iterrows():
                    crop_file_name = save_img_crops(original_img, basename, row_index, table_class_name, rows_boxes[row_index], crop_dir)
                    row_len = len(row)
                    desc = ""
                    amount = ""
                    trans_date = ""
                    debit_amount = ""
                    credit_amount = ""
                    balance_amount = ""
                    other = ""
                    for col_index, col in enumerate(row):
                        if col_mapping[col_index] == "ATOMIC_TBL_COLUMN_DATE":
                            trans_date = parse_date(col)
                        if col_mapping[col_index] == "ATOMIC_TBL_COLUMN_DESC":
                            desc = col
                        if col_mapping[col_index] == "ATOMIC_TBL_COLUMN_AMOUNT":
                            amount = parse_amount(col)
                        if col_mapping[col_index] == "ATOMIC_TBL_COLUMN_DEBIT_AMOUNT":
                            debit_amount = parse_amount(col)
                        if col_mapping[col_index] == "ATOMIC_TBL_COLUMN_CREDIT_AMOUNT":
                            credit_amount = parse_amount(col)
                        if col_mapping[col_index] == "ATOMIC_TBL_COLUMN":
                            other = str(column_header[col_index]).replace("ATOMIC_TBL_COLUMN", "") + str(col)
                            print("other", other)

                    if desc == "":
                        desc = other
                    if table_class_name == "MAPPED_BLOCK_TBL_TABLE_TRANSACTION_DEBIT":
                        debit_amount = amount
                    if table_class_name == "MAPPED_BLOCK_TBL_TABLE_TRANSACTION_CREDIT":
                        credit_amount = amount
                    if not debit_amount and not credit_amount:
                        debit_amount = amount
                        amount = ""

                    final_output.append({'Account Bank Name': bank_name, "Account Holder Name": account_holder_name,
                                         "Account Number": account_number,
                                         "Debit": debit_amount,
                                         "Credit": credit_amount,
                                         "Transaction Note": desc,
                                         # "Deposits": amount,
                                         # "Balance": balance_amount,
                                         "Transaction Date": trans_date,
                                         "Image": crop_file_name,
                                         "Filename": basename
                                         })

                for spur_line in table_spur:
                    if spur_line["CLASS_NAME"] == "MAPPED_GROUP_TABLE_TOTAL":
                        final_output.append({'Account Bank Name': bank_name, "Account Holder Name": account_holder_name,
                                             "Account Number": account_number,
                                             "Transaction Note": desc,
                                             "Balance": spur_line.get("VALUE"),
                                             "Transaction Date": trans_date,
                                             "Filename": basename
                                             })

    final_output_df = pd.DataFrame(final_output, columns=IRS_OUTPUT_COLUMNS)
    # out_file_path = os.path.join(output_dir, basename + ".xlsx")
    # sf = StyleFrame(final_output_df)
    # excel_writer = StyleFrame.ExcelWriter(out_file_path)
    # sf.set_column_width(columns=sf.columns, width=30)
    # sf.to_excel(excel_writer=excel_writer, sheet_name='Bank Statements', row_to_add_filters=0)
    # excel_writer.save()
    return final_output_df  # out_file_path


def post_prcess_crop_text_csv(crop_text_csv_file):
    text_dict = {}
    vendor_info = {}
    with open(crop_text_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['FILENAME'] not in text_dict:
                text_dict[row['FILENAME']] = []
            text_dict[row['FILENAME']].append(
                {"KV": row.get("KV", "") + row.get("Value", ""), "Key": row.get("Key", ""),
                 "Value": row.get("Value"), "type": "KV_GENERAL"})
            if 'BLOCK_NAME' in row and row['BLOCK_NAME'] == "MAPPED_BLOCK_ENTITY_INFO_VND":
                if row.get("GROUP_NAME") == "MAPPED_GROUP_ENTITY_NAME":
                    vendor_info["NAME"] = row.get("VALUE_GENERAL")
                if row.get("GROUP_NAME") == "MAPPED_GROUP_STREET_ADDR":
                    vendor_info["STREET_ADDRESS"] = row.get("VALUE_GENERAL")
                if row.get("GROUP_NAME") == "MAPPED_GROUP_CITY":
                    vendor_info["CITY"] = row.get("VALUE_GENERAL")
                if row.get("GROUP_NAME") == "MAPPED_GROUP_COUNTRY":
                    vendor_info["COUNTRY"] = row.get("VALUE_GENERAL")
                if row.get("GROUP_NAME") == "MAPPED_GROUP_PHONE":
                    vendor_info["PHONE"] = row.get("VALUE_GENERAL")
                if row.get("GROUP_NAME") == "MAPPED_GROUP_STATE":
                    vendor_info["STATE"] = row.get("VALUE_GENERAL")
    return text_dict, vendor_info


def post_prcess_crop_text_csv_old(crop_text_csv_file):
    '''
        post processing for older croptext csv generate by old AI
    :param crop_text_csv_file:
    :return:
    '''
    text_dict = {}
    vendor_info_detected = []
    with open(crop_text_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text_dict[row['FILENAME']] = row
            if row['type'] == "ADDR_VND":
                vendor_info_detected.append(row['Value'])

    return text_dict, vendor_info_detected


def run_kv_text_classifier(text_dict, crops_dict, out_dir, document_type, weight_dir):
    model_info = ai_text_classifier_inference.load_classifier_model(weight_dir, document_type)
    fields = {}
    classifier_results_file = out_dir + "/classifier_results.txt"
    out_fp = open(classifier_results_file, "w")
    out_fp.write("{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".format(
        'Label', 'Probability', 'Key', 'Value', "KeyValue", "Filename"))

    for label in model_info[2]:
        fields[label.strip()] = []
    for filename, rows in text_dict.items():
        for row in rows:
            label, prob = ai_text_classifier_inference.run_inference(
                model_info, row.get("Key", ""), row.get("Value", ""))
            value = row.get("Value", "") or ""
            key = row.get("Key", "") or ""
            fields[label].append((value.strip(), prob, key))
            print("key", key, "value", value, "prob", prob, "label", label)
            out_fp.write("{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".
                         format(label, str(round(prob, 2)), key, value, row["KV"], filename))

    fields = get_fields_from_crops(fields, document_type)
    out_fp.close()
    return fields, crops_dict


def run_text_classifier_inference_old(text_dict, crops_dict, out_dir, document_type, weight_dir):
    model_info = ai_text_classifier_inference.load_classifier_model(weight_dir, document_type)
    classifier_results_file = out_dir + "/classifier_results.txt"
    out_fp = open(classifier_results_file, "w")
    out_fp.write("{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".format(
        'Label', 'Probability', 'Key', 'Value', "KeyValue", "Filename"))
    fields = {}
    for label in model_info[2]:
        fields[label.strip()] = []
        fields['ADDRESS'] = []

    for filename, crop_dict in crops_dict.items():

        crop_info = [None] * len(crop_dict["crops"])

        for i, crop in enumerate(crop_dict["crops"]):
            crop_filename = crop["filename"]
            parts = Path(crop_filename).parts[-2:]
            # file_key = os.path.join(parts[0], parts[1])
            file_key = parts[0] + "/" + parts[1]
            if file_key not in text_dict:
                print(
                    "Warning:",
                    file_key,
                    " does not exist in crop_text. check it")
                out_fp.write(
                    "{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".format(
                        'NA', 'NA', 'NA', 'NA', "NA", file_key))
                continue

            text_rec = text_dict[file_key]
            crop_info[i] = {'label': "KV_OTHER", 'text_rec': text_rec}
            if text_rec['Value'] == '':
                if text_rec['KV'] != '' and text_rec['Key'] != '':
                    value = text_rec['KV'].replace(text_rec['Key'], "")
                else:
                    out_fp.write("{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".format(
                        'NA', 'NA', text_rec["Key"], text_rec["Value"], text_rec["KV"], file_key))
                    continue
            else:
                value = text_rec['Value']

            if text_rec['Key'] == '':
                if text_rec['KV'] != '' and text_rec['Value'] != '':
                    key = text_rec['KV'].replace(text_rec['Value'], "")
                else:
                    out_fp.write("{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".format(
                        'NA', 'NA', text_rec["Key"], text_rec["Value"], text_rec["KV"], file_key))
                    continue
            else:
                key = text_rec['Key']

            if text_rec['type'] == 'KV_ANCHOR':
                key = "ANCHOR " + key

            if text_rec['type'] != 'KV_ADDRESS':
                key_text = key
                value_text = text_rec["Value"]
                label, prob = ai_text_classifier_inference.run_inference(
                    model_info, key_text, value_text)
                if text_rec['type'] == 'KV_ANCHOR':
                    if label == 'ANCHOR_INVD':
                        label = 'KV_INVD'
                    if label == 'ANCHOR_TOTAL':
                        label = "KV_TA"
                    elif label == 'ANCHOR_INVN':
                        label = "KV_INVN"
                    prob = 0.0
            else:
                label = 'ADDRESS'
                prob = 0.0

            crop_info[i] = {'label': label, 'text_rec': text_rec}
            crop['label'] = label

            fields[label].append((value.strip(), prob, text_rec['Key']))

            out_fp.write("{:<8} {:<11} {:<20}    {:<20} {:<40} {:<40}\n".
                         format(label, str(round(prob, 2)),
                                key, value, text_rec["KV"], file_key))
    fields = get_fields_from_crops(fields, document_type)
    out_fp.close()
    return fields, crops_dict

def generate_xlsx_from_dfs(dfs, output_xlsx):
    result_df = pd.concat(dfs, sort=False, ignore_index=True)
    result_df.fillna("", inplace=True)
    sf = StyleFrame(result_df)
    excel_writer = StyleFrame.ExcelWriter(output_xlsx)
    sf.set_column_width(columns=sf.columns, width=30)
    col_width_dict = {
        ("Transaction Note"): 70,
        ("Account Statement Period Start",
         "Account Statement Period End", "Debit", "Credit", "Deposits", "Balance", "Statement Starting Balance",
         "Statement Ending Balance", "Transaction Date", "Transaction Type", "Transaction Vendor Name",
         "Check Number", "Check Payee Name", "Check Payee Address", "Check Amount", "Image", "Filename"): 20,
    }
    sf.set_column_width_dict(col_width_dict=col_width_dict)
    sf.to_excel(excel_writer=excel_writer, sheet_name='Bank statements', row_to_add_filters=0)
    excel_writer.save()
    return output_xlsx


def combine_excel(excel1, excel2):
    df1 = None
    if not os.path.exists(excel1):
        df1 = pd.DataFrame()
    else:
        df1 = pd.read_excel(excel1)
    df2 = pd.read_excel(excel2)

    result_excel = generate_xlsx_from_dfs([df1, df2], excel1)
    return result_excel


def generate_xlsx(result, doc_type, output_xlsx):
    if os.path.isfile(result) and result.endswith('.csv'):
        columns = []
        sheet_name = None
        if doc_type.lower() == "bill":
            sheet_name = "Bills"
            columns = ["File", "Bill Date", "Bill Total", "Amount Due", "Vendor Name", "Vendor Address", "Bill No.",
                       "PO#", "Issued To", "Issue to Address"]
        if doc_type.lower() == "receipt":
            sheet_name = "Receipts"
            columns = ["File", "Bill Date", "Bill Total", "Amount Due", "Vendor Name", "Vendor Address", "Bill No.",
                       "PO#", "Issued To", "Issue to Address"]
        if doc_type.lower() == "bank statements":
            sheet_name = "Bank statements"
            columns = IRS_OUTPUT_COLUMNS
        final_output = {}
        for col in columns:
            final_output[col] = []

        df = pd.read_csv(result)
        cols = [col for col in df]
        for name, values in df.iteritems():
            if name in final_output:
                final_output[name].append(values[0])
        for k, v in final_output.items():
            if k == "File":
                v.append(os.path.basename(result)[:-4])
            elif k not in cols:
                v.append("")
        df = pd.DataFrame.from_dict(final_output, orient='index').transpose()
        df.fillna('', inplace=True)
        if not os.path.isfile(output_xlsx):
            writer = pd.ExcelWriter(output_xlsx, engine='xlsxwriter')
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for idx in range(0, len(columns) + 1):  # loop through all columns
                worksheet.set_column(idx, idx, 25)
            writer.save()

        else:
            print("xlsx already exists ! updatting older xlsx file..")
            try:
                df_excel = pd.read_excel(output_xlsx, sheet_name=sheet_name)
                result_df = pd.concat([df_excel, df], sort=False, ignore_index=True)
            except Exception as e:
                print(e)
                result_df = df
            result_df.fillna('', inplace=True)
            writer = pd.ExcelWriter(output_xlsx, engine='xlsxwriter')
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for idx in range(0, len(columns) + 1):  # loop through all columns
                worksheet.set_column(idx, idx, 25)
            writer.save()
