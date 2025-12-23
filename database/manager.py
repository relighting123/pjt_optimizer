import oracledb
import pandas as pd
import yaml
import os
from datetime import datetime
from config import data_config

class OracleManager:
    def __init__(self, mode=None):
        # YAML 파일에서 설정 로드
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.full_config = yaml.safe_load(f)
        
        # 시스템 모드 결정 (인자 우선, 없으면 config 파일 기준)
        self.mode = mode or self.full_config.get('system_mode', 'local_test')
        
        # 모드별 DB 프로필 로드 (local_test는 DB 접속 정보 불필요)
        if self.mode != 'local_test':
            db_conf = self.full_config.get('database', {}).get(self.mode, {})
            self.user = db_conf.get('user')
            self.password = db_conf.get('password')
            self.dsn = db_conf.get('dsn')
        else:
            self.user = self.password = self.dsn = None

    def _get_connection(self):
        if self.mode == 'local_test':
            return None
        return oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)

    def fetch_inputs(self):
        """
        시스템 모드에 따라 샘플 데이터(local_test) 또는 실데이터(prod/dev) 반환
        """
        if self.mode == 'local_test':
            print(f"[Info] Running in LOCAL_TEST mode. Returning sample data.")
            return (
                data_config.DEMAND, 
                data_config.EQUIPMENT_MODELS, 
                data_config.PROCESS_CONFIG, 
                data_config.WIP,
                data_config.EQP_WIP,
                data_config.TOOLS
            )

        try:
            print(f"[Info] Fetching inputs from Oracle ({self.mode} DB)...")
            with self._get_connection() as conn:
                # 1. 수요 데이터
                demand_query = "SELECT PRODUCT_ID, DEMAND_QTY FROM TB_PRODUCTION_PLAN"
                demand_df = pd.read_sql(demand_query, conn)
                demands = dict(zip(demand_df['PRODUCT_ID'], demand_df['DEMAND_QTY']))

                # 2. 장비 데이터
                eqp_query = "SELECT MODEL_ID, UNIT_ID FROM TB_EQUIPMENT_MASTER"
                eqp_df = pd.read_sql(eqp_query, conn)
                equipment_models = eqp_df.groupby('MODEL_ID')['UNIT_ID'].apply(list).to_dict()

                # 3. 공정 설정 (초 -> 분 변환)
                proc_query = "SELECT PRODUCT_ID, OPER_ID, MODEL_ID, CYCLE_TIME FROM TB_PROCESS_STANDARD"
                proc_df = pd.read_sql(proc_query, conn)
                process_config = {(row['PRODUCT_ID'], row['OPER_ID'], row['MODEL_ID']): row['CYCLE_TIME'] / 60.0 for _, row in proc_df.iterrows()}

                # 4. 재공량 (Input WIP)
                wip_query = "SELECT PRODUCT_ID, OPER_ID, WIP_QTY FROM TB_WIP_STATUS"
                wip_df = pd.read_sql(wip_query, conn)
                wip = {(row['PRODUCT_ID'], row['OPER_ID']): row['WIP_QTY'] for _, row in wip_df.iterrows()}

                # 5. 장비별 현재 작업 재공 (Equipment WIP) - 초 -> 분 변환
                eqw_query = "SELECT EQP_ID, PROD_ID, OPER_ID, END_TIME FROM TB_EQP_WIP"
                eqw_df = pd.read_sql(eqw_query, conn)
                eqp_wip = {}
                now = datetime.now()
                for _, row in eqw_df.iterrows():
                    offset_min = (row['END_TIME'] - now).total_seconds() / 60.0
                    eqp_wip[row['EQP_ID']] = {
                        'Product': row['PROD_ID'],
                        'Operation': row['OPER_ID'],
                        'End_Time_Offset': max(0, offset_min)
                    }

                # 6. 연간/공정별 툴 수량 (Tool Constraints)
                tool_query = "SELECT PRODUCT_ID, OPER_ID, TOOL_QTY FROM TB_TOOL_MASTER"
                tool_df = pd.read_sql(tool_query, conn)
                tools = {(row['PRODUCT_ID'], row['OPER_ID']): row['TOOL_QTY'] for _, row in tool_df.iterrows()}

                return demands, equipment_models, process_config, wip, eqp_wip, tools

        except Exception as e:
            print(f"Failed to fetch inputs from Oracle ({self.mode}): {e}")
            return None, None, None, None, None, None

        except Exception as e:
            print(f"Failed to fetch inputs from Oracle ({self.mode}): {e}")
            return None, None, None, None

    def upload_results(self, df):
        if self.mode == 'local_test' or df is None or df.empty:
            print(f"[Info] skipping upload (Mode: {self.mode})")
            return

        rule_timekey = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "INSERT INTO PRODUCTION_RESULTS (RULE_TIMEKEY, EQP_ID, START_TIME, END_TIME, PROD_ID, OPER_ID) VALUES (:1, :2, :3, :4, :5, :6)"
                    data = []
                    for _, row in df.iterrows():
                        data.append((rule_timekey, row['Unit'], row['Start_Time'], row['End_Time'], row['Product'], row['Operation']))
                    cursor.executemany(sql, data)
                    conn.commit()
            print(f"Successfully uploaded {len(df)} rows to Oracle ({self.mode}).")
        except Exception as e:
            print(f"Failed to upload to Oracle ({self.mode}): {e}")
