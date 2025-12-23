import oracledb
import pandas as pd
import yaml
import os
from datetime import datetime

class OracleManager:
    def __init__(self, user=None, password=None, dsn=None):
        # YAML 파일에서 기본값 로드 시도
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        file_config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                file_config = full_config.get('database', {})

        self.conn_params = {
            "user": user or file_config.get('user', 'ADMIN'),
            "password": password or file_config.get('password', ''),
            "dsn": dsn or file_config.get('dsn', 'localhost:1521/xe')
        }

    def _get_connection(self):
        return oracledb.connect(**self.conn_params)

    def upload_results(self, df):
        """
        데이터 결과 적재: RULE_TIMEKEY, EQP_ID, START_TIME, END_TIME, PROD_ID, OPER_ID
        """
        if df.empty:
            return

        rule_timekey = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 컬럼명 매핑 및 데이터 정리
        upload_df = pd.DataFrame()
        upload_df['RULE_TIMEKEY'] = [rule_timekey] * len(df)
        upload_df['EQP_ID'] = df['Unit']
        upload_df['START_TIME'] = df['Start_Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        upload_df['END_TIME'] = df['End_Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        upload_df['PROD_ID'] = df['Product']
        upload_df['OPER_ID'] = df['Operation']

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 예시 테이블: PRODUCTION_RESULTS
                    sql = """
                    INSERT INTO PRODUCTION_RESULTS 
                    (RULE_TIMEKEY, EQP_ID, START_TIME, END_TIME, PROD_ID, OPER_ID) 
                    VALUES (:1, :2, :3, :4, :5, :6)
                    """
                    data = upload_df.values.tolist()
                    cursor.executemany(sql, data)
                    conn.commit()
            print(f"Successfully uploaded {len(upload_df)} rows to Oracle.")
        except Exception as e:
            print(f"Failed to upload to Oracle: {e}")

    def fetch_inputs(self):
        """
        Oracle DB에서 기초 데이터를 가져오는 샘플 코드입니다.
        아래 쿼리와 매핑 로직을 실제 운영 테이블 전문에 맞게 수정하여 사용하세요.
        """
        try:
            with self._get_connection() as conn:
                # 1. 제품별 계획량 (Demand)
                # 쿼리 예시: 제품코드와 수량을 가져옵니다.
                demand_query = """
                SELECT PRODUCT_ID, DEMAND_QTY 
                FROM TB_PRODUCTION_PLAN 
                WHERE PLAN_DATE = TO_CHAR(SYSDATE, 'YYYYMMDD')
                """
                demand_df = pd.read_sql(demand_query, conn)
                demands = dict(zip(demand_df['PRODUCT_ID'], demand_df['DEMAND_QTY']))

                # 2. 장비 모델 및 호기 구성 (Equipment)
                # 쿼리 예시: 모델별로 속한 장비 호기 리스트를 가져옵니다.
                eqp_query = "SELECT MODEL_ID, UNIT_ID FROM TB_EQUIPMENT_MASTER WHERE USE_YN = 'Y'"
                eqp_df = pd.read_sql(eqp_query, conn)
                equipment_models = eqp_df.groupby('MODEL_ID')['UNIT_ID'].apply(list).to_dict()

                # 3. 공정 시간 및 장비 모델 혼용 정보 (Process Config)
                # 쿼리 예시: (제품, 공정, 모델)별 Unit Time(초) 정보를 가져옵니다.
                proc_query = """
                SELECT PRODUCT_ID, OPER_ID, MODEL_ID, CYCLE_TIME 
                FROM TB_PROCESS_STANDARD 
                """
                proc_df = pd.read_sql(proc_query, conn)
                process_config = {}
                for _, row in proc_df.iterrows():
                    key = (row['PRODUCT_ID'], row['OPER_ID'], row['MODEL_ID'])
                    process_config[key] = row['CYCLE_TIME']

                print(f"Successfully fetched inputs from Oracle at {datetime.now()}")
                return demands, equipment_models, process_config

        except Exception as e:
            print(f"Failed to fetch inputs from Oracle: {e}")
            # 에러 발생 시 None을 반환하여 호출부에서 예외 처리하게 함
            return None, None, None
