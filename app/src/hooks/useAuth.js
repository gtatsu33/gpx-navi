import { useEffect, useState } from 'react'
import { getSession, onAuthStateChange, sendMagicLink, signOut } from '../lib/supabase.js'

/**
 * 招待制ログイン（Supabase Auth マジックリンク）のセッション管理。
 * spec.txt 19章・3-4章／implement.txt 13章。
 * マジックリンククリック後のセッション復元はsupabase-jsの
 * detectSessionInUrl（デフォルト有効）が自動処理するため、ここでは
 * getSession/onAuthStateChangeを購読するだけでよい。
 */
export function useAuth() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    getSession().then((session) => {
      if (cancelled) return
      setUser(session?.user ?? null)
      setLoading(false)
    })
    const unsubscribe = onAuthStateChange((session) => {
      setUser(session?.user ?? null)
      setLoading(false)
    })
    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  return { user, loading, sendMagicLink, signOut }
}
