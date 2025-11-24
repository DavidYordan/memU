import asyncio
import uuid
from typing import Any

import asyncpg

from memu.memory.repo import InMemoryStore
from memu.models import Resource, MemoryItem, MemoryCategory, CategoryItem


class PostgresClient:
    def __init__(self, dsn: str, embed_dim: int = 1536):
        self.dsn = dsn
        self.pool = None
        self.embed_dim = embed_dim

    def initialize(self):
        return None

    async def ensure_pool(self):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=5)

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def upsert_resource(self, res: Resource):
        await self.ensure_pool()
        if self.pool is None:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO resources (id, url, local_path, modality, caption, embedding, embed_model, embed_dim)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                  url=EXCLUDED.url,
                  local_path=EXCLUDED.local_path,
                  modality=EXCLUDED.modality,
                  caption=EXCLUDED.caption,
                  embedding=EXCLUDED.embedding,
                  embed_model=EXCLUDED.embed_model,
                  embed_dim=EXCLUDED.embed_dim
                """,
                res.id,
                res.url,
                res.local_path,
                res.modality,
                res.caption,
                res.embedding,
                None,
                self.embed_dim,
            )

    async def upsert_category(self, cat: MemoryCategory):
        await self.ensure_pool()
        if self.pool is None:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO categories (id, name, description, summary, embedding, embed_model, embed_dim)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE SET
                  name=EXCLUDED.name,
                  description=EXCLUDED.description,
                  summary=EXCLUDED.summary,
                  embedding=EXCLUDED.embedding,
                  embed_model=EXCLUDED.embed_model,
                  embed_dim=EXCLUDED.embed_dim
                """,
                cat.id,
                cat.name,
                cat.description,
                cat.summary,
                cat.embedding,
                None,
                self.embed_dim,
            )

    async def upsert_item(self, item: MemoryItem):
        await self.ensure_pool()
        if self.pool is None:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO items (id, summary, embedding, created_at)
                VALUES ($1, $2, $3, now())
                ON CONFLICT (id) DO UPDATE SET
                  summary=EXCLUDED.summary,
                  embedding=EXCLUDED.embedding
                """,
                item.id,
                item.summary,
                item.embedding,
            )

    async def upsert_relation(self, rel: CategoryItem):
        await self.ensure_pool()
        if self.pool is None:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO relations (id, item_id, category_id)
                VALUES ($1, $2, $3)
                ON CONFLICT (id) DO NOTHING
                """,
                str(uuid.uuid4()),
                rel.item_id,
                rel.category_id,
            )

    async def load_into_store(self, store: InMemoryStore):
        await self.ensure_pool()
        if self.pool is None:
            return False
        async with self.pool.acquire() as conn:
            cats = await conn.fetch("SELECT id, name, description, summary, embedding FROM categories")
            for r in cats:
                cat = MemoryCategory(
                    id=str(r[0]),
                    name=str(r[1] or ""),
                    description=str(r[2] or ""),
                    summary=str(r[3]) if r[3] is not None else None,
                    embedding=list(r[4]) if r[4] is not None else None,
                )
                store.categories[cat.id] = cat
            ress = await conn.fetch("SELECT id, url, local_path, modality, caption, embedding FROM resources")
            for r in ress:
                res = Resource(
                    id=str(r[0]),
                    url=str(r[1] or ""),
                    local_path=str(r[2] or ""),
                    modality=str(r[3] or ""),
                    caption=str(r[4]) if r[4] is not None else None,
                    embedding=list(r[5]) if r[5] is not None else None,
                )
                store.resources[res.id] = res
            its = await conn.fetch("SELECT id, summary, embedding FROM items")
            for r in its:
                item = MemoryItem(
                    id=str(r[0]),
                    resource_id="",
                    memory_type="knowledge",
                    summary=str(r[1] or ""),
                    embedding=list(r[2]) if r[2] is not None else None,
                )
                store.items[item.id] = item
            rels = await conn.fetch("SELECT item_id, category_id FROM relations")
            for r in rels:
                store.relations.append(CategoryItem(item_id=str(r[0]), category_id=str(r[1])))
        return True


class PersistentStoreProxy:
    def __init__(self, store: InMemoryStore, db: PostgresClient | None):
        self.store = store
        self.db = db

    def create_resource(self, *, url: str, modality: str, local_path: str) -> Resource:
        res = self.store.create_resource(url=url, modality=modality, local_path=local_path)
        return res

    def get_or_create_category(self, *, name: str, description: str, embedding: list[float] | None) -> MemoryCategory:
        cat = self.store.get_or_create_category(name=name, description=description, embedding=embedding)
        return cat

    def create_item(self, *, resource_id: str, memory_type: str, summary: str, embedding: list[float] | None) -> MemoryItem:
        item = self.store.create_item(resource_id=resource_id, memory_type=memory_type, summary=summary, embedding=embedding)
        return item

    def link_item_category(self, item_id: str, cat_id: str) -> CategoryItem:
        rel = self.store.link_item_category(item_id, cat_id)
        return rel

    async def persist_resource(self, res: Resource):
        if self.db:
            await self.db.upsert_resource(res)

    async def persist_category(self, cat: MemoryCategory):
        if self.db:
            await self.db.upsert_category(cat)

    async def persist_item(self, item: MemoryItem):
        if self.db:
            await self.db.upsert_item(item)

    async def persist_relation(self, rel: CategoryItem):
        if self.db:
            await self.db.upsert_relation(rel)

    async def load_all(self):
        if self.db:
            return await self.db.load_into_store(self.store)
        return False
