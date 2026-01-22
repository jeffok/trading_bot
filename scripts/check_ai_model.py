#!/usr/bin/env python3
"""检查AI模型状态的工具脚本"""
import json
import sys
from shared.db import PostgreSQL
from shared.config.loader import load_settings
from shared.ai.model_store import load_current_model_blob

def check_ai_model_status():
    """检查AI模型的状态"""
    settings = load_settings()
    db = PostgreSQL(settings.postgres_url)
    
    try:
        # 1. 检查 ai_models 表
        print("=" * 80)
        print("1. 检查 ai_models 表中的模型")
        print("=" * 80)
        
        rows = db.fetch_all("""
            SELECT 
                id,
                model_name,
                version,
                is_current,
                created_at,
                LENGTH(blob) as blob_size
            FROM ai_models
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        if not rows:
            print("❌ ai_models 表中没有模型记录")
        else:
            print(f"找到 {len(rows)} 条模型记录：")
            print("-" * 80)
            print("id | model_name | version | is_current | created_at | blob_size")
            print("-" * 80)
            for row in rows:
                print(f"{row['id']} | {row['model_name']} | {row['version']} | {row['is_current']} | {row['created_at']} | {row['blob_size'] or 0} bytes")
        
        # 2. 检查当前模型（is_current=TRUE）
        print("\n" + "=" * 80)
        print("2. 检查当前模型（is_current=TRUE）")
        print("=" * 80)
        
        current_rows = db.fetch_all("""
            SELECT 
                model_name,
                version,
                created_at
            FROM ai_models
            WHERE is_current=TRUE
            ORDER BY created_at DESC
        """)
        
        if not current_rows:
            print("❌ 没有标记为 is_current=TRUE 的模型")
        else:
            print(f"找到 {len(current_rows)} 个当前模型：")
            for row in current_rows:
                print(f"  - {row['model_name']} (version: {row['version']}, created_at: {row['created_at']})")
        
        # 3. 加载并检查模型内容
        print("\n" + "=" * 80)
        print("3. 检查模型训练状态（从 blob 字段解析）")
        print("=" * 80)
        
        model_name = settings.ai_model_key if hasattr(settings, 'ai_model_key') else 'default'
        blob_dict = load_current_model_blob(db, model_name=model_name)
        
        if not blob_dict:
            print(f"❌ 无法加载模型 '{model_name}' 的 blob 数据")
            print("   可能原因：")
            print("   1. 模型尚未保存到数据库")
            print("   2. blob 字段为空")
            print("   3. blob 数据格式错误")
        else:
            print(f"✅ 成功加载模型 '{model_name}' 的 blob 数据")
            print("-" * 80)
            print("模型信息：")
            print(f"  - dim (特征维度): {blob_dict.get('dim', 'N/A')}")
            print(f"  - lr (学习率): {blob_dict.get('lr', 'N/A')}")
            print(f"  - l2 (L2正则化): {blob_dict.get('l2', 'N/A')}")
            print(f"  - bias: {blob_dict.get('bias', 'N/A')}")
            print(f"  - seen (训练样本数): {blob_dict.get('seen', 0)}")
            print(f"  - version: {blob_dict.get('version', 'N/A')}")
            print(f"  - impl: {blob_dict.get('impl', 'online_lr (默认)')}")
            
            seen = blob_dict.get('seen', 0)
            if seen == 0:
                print("\n⚠️  警告：模型尚未训练（seen=0），所有AI评分将为50.0")
                print("   模型需要在实盘交易中通过 partial_fit 进行训练")
            else:
                print(f"\n✅ 模型已训练，训练样本数: {seen}")
                
                # 检查权重是否全为0
                w = blob_dict.get('w', [])
                if w and all(x == 0.0 for x in w) and blob_dict.get('bias', 0.0) == 0.0:
                    print("⚠️  警告：模型权重全为0，评分仍将为50.0")
                else:
                    print("✅ 模型权重非零，可以产生真实的AI评分")
        
        # 4. 检查 system_config 中的模型
        print("\n" + "=" * 80)
        print("4. 检查 system_config 表中的模型配置")
        print("=" * 80)
        
        config_rows = db.fetch_all(
            """
            SELECT 
                key,
                LENGTH(value) as value_length,
                updated_at
            FROM system_config
            WHERE key LIKE %s OR key LIKE %s
            ORDER BY updated_at DESC
            """,
            ('%AI_MODEL%', '%ai_model%')
        )
        
        if not config_rows:
            print("❌ system_config 表中没有AI模型相关的配置")
        else:
            print(f"找到 {len(config_rows)} 条相关配置：")
            for row in config_rows:
                print(f"  - {row['key']} (长度: {row['value_length']} bytes, 更新时间: {row['updated_at']})")
        
        # 5. 检查 runtime_config
        print("\n" + "=" * 80)
        print("5. 检查运行时配置")
        print("=" * 80)
        
        from shared.domain.runtime_config import RuntimeConfig
        runtime_cfg = RuntimeConfig.load(db)
        
        print(f"  - ai_enabled: {runtime_cfg.ai_enabled}")
        print(f"  - ai_lr: {runtime_cfg.ai_lr}")
        print(f"  - ai_model_key: {getattr(settings, 'ai_model_key', 'N/A')}")
        print(f"  - ai_model_impl: {getattr(settings, 'ai_model_impl', 'N/A')}")
        
        if not runtime_cfg.ai_enabled:
            print("\n⚠️  警告：AI未启用（ai_enabled=False），回测时将使用默认评分50.0")
        
        print("\n" + "=" * 80)
        print("检查完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ 检查失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(check_ai_model_status())
