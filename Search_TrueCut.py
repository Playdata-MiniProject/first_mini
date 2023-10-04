import csv

# 원본 CSV 파일 경로
csv_file_path = './open/NASDAQ_DT_FC_STK_QUT.csv'
# 새로운 CSV 파일 경로
output_csv_path = 'NASDAQ_DT_FC_STK_QUT_TrueCut.csv'  # 수정된 결과를 저장할 파일

def remove_true_and_create_new_csv(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', newline='') as csv_file, \
         open(output_path, 'w', encoding='utf-8', newline='') as output_csv_file:
        
        csv_reader = csv.reader(csv_file)
        csv_writer = csv.writer(output_csv_file)
        
        for row in csv_reader:
            if not any(value.strip().lower() == "true" for value in row):
                # 'True'를 포함하지 않는 행만을 새로운 파일에 쓰기
                csv_writer.writerow(row)

    print(f"'True'가 포함된 행을 삭제하고 새로운 CSV 파일이 생성되었습니다. 경로: {output_path}")

remove_true_and_create_new_csv(csv_file_path, output_csv_path)
