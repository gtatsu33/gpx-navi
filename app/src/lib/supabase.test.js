import { describe, expect, it, vi } from 'vitest'
import { downloadGpx, getSession, listRoutes, onAuthStateChange, sendMagicLink, signOut, uploadGpx } from './supabase.js'

function makeMockClient({ uploadError = null, insertError = null, removeSpy = vi.fn() } = {}) {
  return {
    storage: {
      from: () => ({
        upload: vi.fn().mockResolvedValue({ error: uploadError }),
        remove: removeSpy,
        download: vi.fn(),
      }),
    },
    from: () => ({
      insert: vi.fn().mockResolvedValue({ error: insertError }),
    }),
  }
}

describe('uploadGpx', () => {
  it('正常系: fileKeyを返す', async () => {
    const client = makeMockClient()
    const result = await uploadGpx('<gpx/>', 'route1', { displayName: 'ルート1', client })
    expect(result).toEqual({ ok: true, fileKey: 'route1.gpx', errorType: null })
  })

  it('Storageアップロードで重複エラー時はfile_key_dupを返す', async () => {
    const client = makeMockClient({ uploadError: { message: 'The resource already exists' } })
    const result = await uploadGpx('<gpx/>', 'route1', { displayName: 'x', client })
    expect(result.ok).toBe(false)
    expect(result.errorType).toBe('file_key_dup')
  })

  it('DB登録失敗時はStorageをロールバックする', async () => {
    const removeSpy = vi.fn()
    const client = makeMockClient({ insertError: { message: 'duplicate key value violates unique constraint "route_files_display_name_key"' }, removeSpy })
    const result = await uploadGpx('<gpx/>', 'route1', { displayName: 'x', client })
    expect(result.ok).toBe(false)
    expect(result.errorType).toBe('display_name_dup')
    expect(removeSpy).toHaveBeenCalledWith(['route1.gpx'])
  })
})

describe('listRoutes', () => {
  it('正常系: routesを返す', async () => {
    const client = {
      from: () => ({
        select: () => ({
          order: vi.fn().mockResolvedValue({ data: [{ file_key: 'a.gpx' }], error: null }),
        }),
      }),
    }
    const result = await listRoutes({ client })
    expect(result).toEqual({ ok: true, routes: [{ file_key: 'a.gpx' }] })
  })

  it('エラー時はok:falseを返す', async () => {
    const client = {
      from: () => ({
        select: () => ({
          order: vi.fn().mockResolvedValue({ data: null, error: { message: 'network down' } }),
        }),
      }),
    }
    const result = await listRoutes({ client })
    expect(result).toEqual({ ok: false, error: 'network down' })
  })
})

describe('downloadGpx', () => {
  it('正常系: テキスト内容を返す', async () => {
    const client = {
      storage: {
        from: () => ({
          download: vi.fn().mockResolvedValue({ data: { text: async () => '<gpx/>' }, error: null }),
        }),
      },
    }
    const result = await downloadGpx('a.gpx', { client })
    expect(result).toEqual({ ok: true, content: '<gpx/>' })
  })
})

describe('招待制ログイン（Supabase Auth マジックリンク）', () => {
  it('sendMagicLink: 正常系はok:trueを返す', async () => {
    const signInWithOtp = vi.fn().mockResolvedValue({ error: null })
    const client = { auth: { signInWithOtp } }
    const result = await sendMagicLink('a@example.com', { client })
    expect(result).toEqual({ ok: true })
    expect(signInWithOtp).toHaveBeenCalledWith({
      email: 'a@example.com',
      options: { emailRedirectTo: expect.any(String) },
    })
  })

  it('sendMagicLink: エラー時はok:falseを返す', async () => {
    const client = { auth: { signInWithOtp: vi.fn().mockResolvedValue({ error: { message: 'invalid email' } }) } }
    const result = await sendMagicLink('bad', { client })
    expect(result).toEqual({ ok: false, error: 'invalid email' })
  })

  it('getSession: セッションを返す', async () => {
    const session = { user: { email: 'a@example.com' } }
    const client = { auth: { getSession: vi.fn().mockResolvedValue({ data: { session } }) } }
    const result = await getSession({ client })
    expect(result).toBe(session)
  })

  it('onAuthStateChange: コールバックを購読し、unsubscribeで解除できる', () => {
    const unsubscribe = vi.fn()
    const client = {
      auth: {
        onAuthStateChange: vi.fn((cb) => {
          cb('SIGNED_IN', { user: { email: 'a@example.com' } })
          return { data: { subscription: { unsubscribe } } }
        }),
      },
    }
    const callback = vi.fn()
    const unsub = onAuthStateChange(callback, { client })
    expect(callback).toHaveBeenCalledWith({ user: { email: 'a@example.com' } })
    unsub()
    expect(unsubscribe).toHaveBeenCalled()
  })

  it('signOut: client.auth.signOutを呼ぶ', async () => {
    const signOutFn = vi.fn().mockResolvedValue({})
    const client = { auth: { signOut: signOutFn } }
    await signOut({ client })
    expect(signOutFn).toHaveBeenCalled()
  })
})
