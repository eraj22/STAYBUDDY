import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';

class ParentProfileScreen extends StatefulWidget {
  const ParentProfileScreen({super.key});

  @override
  State<ParentProfileScreen> createState() =>
      _ParentProfileScreenState();
}

class _ParentProfileScreenState
    extends State<ParentProfileScreen> {
  late VideoPlayerController _videoController;
  String parentName = 'Mr. Raza Ahmed';
  String parentEmail = 'raza.parent@gmail.com';
  String parentPhone = '0321-1234567';

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);
  static const Color green = Color(0xFF35EE96);
  static const Color red = Color(0xFFFF6B6B);

  bool attendanceNotifications = true;
  bool feeReminders = true;
  bool announcements = true;
  bool emergencyAlerts = true;
  bool locationUpdates = true;

  @override
  void initState() {
    super.initState();

    _videoController = VideoPlayerController.asset(
      'assets/images/background.mp4',
    )
      ..initialize().then((_) {
        if (!mounted) return;

        _videoController
          ..setLooping(true)
          ..setVolume(0)
          ..play();

        setState(() {});
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  void _showComingSoon(String title) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$title will be connected later.'),
        behavior: SnackBarBehavior.floating,
        backgroundColor: darkTeal,
      ),
    );
  }

  void _showLogoutDialog() {
    showDialog(
      context: context,
      useRootNavigator: false,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: const Color(0xFF003434),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(22),
          ),
          title: Text(
            'Logout',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontWeight: FontWeight.w800,
            ),
          ),
          content: Text(
            'Are you sure you want to logout from your Parent account?',
            style: GoogleFonts.dmSans(
              color: Colors.white60,
              fontSize: 12,
              height: 1.45,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
              },
              child: Text(
                'Cancel',
                style: GoogleFonts.dmSans(
                  color: Colors.white54,
                ),
              ),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);

                Navigator.of(
                  context,
                  rootNavigator: true,
                ).pushNamedAndRemoveUntil(
                  '/role',
                  (route) => false,
                );
              },
              child: Text(
                'Logout',
                style: GoogleFonts.dmSans(
                  color: red,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  void _showEditProfileDialog() {
    final nameController = TextEditingController(text: parentName);
    final emailController = TextEditingController(text: parentEmail);
    final phoneController = TextEditingController(text: parentPhone);

    showDialog(
      context: context,
      useRootNavigator: false,
      builder: (dialogContext) => AlertDialog(
        backgroundColor: const Color(0xFF063E40),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(22),
        ),
        title: Text(
          'Edit Profile',
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontWeight: FontWeight.w800,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Full Name',
              style: GoogleFonts.dmSans(
                color: Colors.white70,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 7),
            TextField(
              controller: nameController,
              decoration: const InputDecoration(
                hintText: 'Enter your full name',
                prefixIcon: Icon(Icons.person_outline_rounded),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Email Address',
              style: GoogleFonts.dmSans(
                color: Colors.white70,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 7),
            TextField(
              controller: emailController,
              keyboardType: TextInputType.emailAddress,
              decoration: const InputDecoration(
                hintText: 'Enter your email address',
                prefixIcon: Icon(Icons.email_outlined),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Phone Number',
              style: GoogleFonts.dmSans(
                color: Colors.white70,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 7),
            TextField(
              controller: phoneController,
              keyboardType: TextInputType.phone,
              decoration: const InputDecoration(
                hintText: 'Enter your phone number',
                prefixIcon: Icon(Icons.phone_outlined),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            style: TextButton.styleFrom(
              foregroundColor: mint,
            ),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              if (nameController.text.trim().isEmpty ||
                  emailController.text.trim().isEmpty ||
                  phoneController.text.trim().isEmpty) {
                return;
              }
              setState(() {
                parentName = nameController.text.trim();
                parentEmail = emailController.text.trim();
                parentPhone = phoneController.text.trim();
              });
              Navigator.pop(dialogContext);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      body: Stack(
        children: [
          _buildBackground(),

          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(.46),
            ),
          ),

          SafeArea(
            child: Column(
              children: [
                _buildTopBar(),

                Expanded(
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(
                      18,
                      10,
                      18,
                      32,
                    ),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(
                          maxWidth: 600,
                        ),
                        child: Column(
                          crossAxisAlignment:
                              CrossAxisAlignment.start,
                          children: [
                            _buildProfileHeader(),

                            const SizedBox(height: 22),

                            _sectionTitle(
                              'Account Information',
                              'Your registered Parent account details',
                            ),

                            const SizedBox(height: 12),

                            _buildAccountCard(),

                            const SizedBox(height: 22),

                            _sectionTitle(
                              'Linked Child',
                              'Student connected with this Parent account',
                            ),

                            const SizedBox(height: 12),

                            _buildLinkedChildCard(),

                            const SizedBox(height: 22),

                            _sectionTitle(
                              'Notifications',
                              'Choose which alerts you want to receive',
                            ),

                            const SizedBox(height: 12),

                            _buildNotificationCard(),

                            const SizedBox(height: 18),

                            _buildLogoutButton(),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackground() {
    if (!_videoController.value.isInitialized) {
      return const Positioned.fill(
        child: ColoredBox(
          color: background,
        ),
      );
    }

    return Positioned.fill(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _videoController.value.size.width,
          height: _videoController.value.size.height,
          child: VideoPlayer(
            _videoController,
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        18,
        8,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(
              Icons.arrow_back_ios_new_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),

          const SizedBox(width: 4),

          Expanded(
            child: Text(
              'Profile & Settings',
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),

          IconButton(
            tooltip: 'Edit profile',
            onPressed: _showEditProfileDialog,
            icon: const Icon(
              Icons.edit_rounded,
              color: mint,
              size: 21,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProfileHeader() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF00645E).withOpacity(.96),
            const Color(0xFF003B3B).withOpacity(.96),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: mint.withOpacity(.20),
        ),
      ),
      child: Column(
        children: [
          Container(
            width: 82,
            height: 82,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withOpacity(.09),
              border: Border.all(
                color: mint,
                width: 1.5,
              ),
            ),
            child: const Icon(
              Icons.person_rounded,
              color: Colors.white,
              size: 50,
            ),
          ),

          const SizedBox(height: 14),

          Text(
            parentName,
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 21,
              fontWeight: FontWeight.w800,
            ),
          ),

          const SizedBox(height: 4),

          Text(
            'Parent Account',
            style: GoogleFonts.dmSans(
              color: mint,
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),

          const SizedBox(height: 12),

          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 12,
              vertical: 7,
            ),
            decoration: BoxDecoration(
              color: green.withOpacity(.10),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 7,
                  height: 7,
                  decoration: const BoxDecoration(
                    color: green,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 6),
                Text(
                  'Account Active',
                  style: GoogleFonts.dmSans(
                    color: green,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(
    String title,
    String subtitle,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            color: Colors.white38,
            fontSize: 11,
          ),
        ),
      ],
    );
  }

  Widget _buildAccountCard() {
    return _glassCard(
      child: Column(
        children: [
          _ProfileInfoRow(
            icon: Icons.person_outline_rounded,
            label: 'Full Name',
            value: parentName,
          ),
          _ProfileInfoRow(
            icon: Icons.email_outlined,
            label: 'Email',
            value: parentEmail,
          ),
          _ProfileInfoRow(
            icon: Icons.phone_outlined,
            label: 'Phone',
            value: parentPhone,
          ),
          _ProfileInfoRow(
            icon: Icons.badge_outlined,
            label: 'Account Type',
            value: 'Parent',
          ),
          _ProfileInfoRow(
            icon: Icons.admin_panel_settings_outlined,
            label: 'Credentials',
            value: 'Generated by Hostel Owner',
            isLast: true,
          ),
        ],
      ),
    );
  }

  Widget _buildLinkedChildCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: mint.withOpacity(.11),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: teal.withOpacity(.12),
              border: Border.all(
                color: mint.withOpacity(.25),
              ),
            ),
            child: const Icon(
              Icons.person_rounded,
              color: Colors.white,
              size: 35,
            ),
          ),

          const SizedBox(width: 13),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  'Ali Raza',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 14.5,
                    fontWeight: FontWeight.w800,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  'FAST National University',
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: mint,
                    fontSize: 10.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  'Room A-204 • Bed 2',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white38,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          const Icon(
            Icons.link_rounded,
            color: green,
            size: 20,
          ),
        ],
      ),
    );
  }

  Widget _buildNotificationCard() {
    return _glassCard(
      child: Column(
        children: [
          _notificationSwitch(
            icon: Icons.login_rounded,
            title: 'IN / OUT Notifications',
            subtitle: 'Receive hostel entry and exit alerts',
            value: attendanceNotifications,
            onChanged: (value) {
              setState(() {
                attendanceNotifications = value;
              });
            },
          ),

          _divider(),

          _notificationSwitch(
            icon: Icons.payments_outlined,
            title: 'Fee Reminders',
            subtitle: 'Receive reminders before fees are due',
            value: feeReminders,
            onChanged: (value) {
              setState(() {
                feeReminders = value;
              });
            },
          ),

          _divider(),

          _notificationSwitch(
            icon: Icons.campaign_outlined,
            title: 'Announcements',
            subtitle: 'Receive new hostel announcements',
            value: announcements,
            onChanged: (value) {
              setState(() {
                announcements = value;
              });
            },
          ),

          _divider(),

          _notificationSwitch(
            icon: Icons.emergency_outlined,
            title: 'Emergency Alerts',
            subtitle: 'Receive urgent alerts from your child',
            value: emergencyAlerts,
            onChanged: (value) {
              setState(() {
                emergencyAlerts = value;
              });
            },
          ),

          _divider(),

          _notificationSwitch(
            icon: Icons.location_on_outlined,
            title: 'Location Updates',
            subtitle: 'Receive child location status updates',
            value: locationUpdates,
            onChanged: (value) {
              setState(() {
                locationUpdates = value;
              });
            },
          ),
        ],
      ),
    );
  }

  Widget _notificationSwitch({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: 10,
      ),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: mint.withOpacity(.07),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              icon,
              color: mint,
              size: 20,
            ),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),

                const SizedBox(height: 2),

                Text(
                  subtitle,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white38,
                    fontSize: 9.8,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),

          Switch(
            value: value,
            onChanged: onChanged,
            activeColor: mint,
            activeTrackColor: teal.withOpacity(.55),
            inactiveThumbColor: Colors.white38,
            inactiveTrackColor: Colors.white12,
          ),
        ],
      ),
    );
  }

  Widget _buildSecurityCard() {
    return _glassCard(
      child: Column(
        children: [
          _settingsItem(
            icon: Icons.lock_outline_rounded,
            title: 'Change Password',
            subtitle: 'Update your Parent account password',
            onTap: () =>
                _showComingSoon('Change Password'),
          ),

          _divider(),

          _settingsItem(
            icon: Icons.privacy_tip_outlined,
            title: 'Privacy & Security',
            subtitle: 'Manage account security settings',
            onTap: () =>
                _showComingSoon('Privacy & Security'),
          ),

          _divider(),

          _settingsItem(
            icon: Icons.help_outline_rounded,
            title: 'Help & Support',
            subtitle: 'Contact hostel administration for help',
            onTap: () =>
                _showComingSoon('Help & Support'),
          ),
        ],
      ),
    );
  }

  Widget _settingsItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding: const EdgeInsets.symmetric(
            vertical: 12,
          ),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: mint.withOpacity(.07),
                  borderRadius:
                      BorderRadius.circular(13),
                ),
                child: Icon(
                  icon,
                  color: mint,
                  size: 20,
                ),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow:
                          TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 11.5,
                        fontWeight:
                            FontWeight.w700,
                      ),
                    ),

                    const SizedBox(height: 2),

                    Text(
                      subtitle,
                      maxLines: 2,
                      overflow:
                          TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white38,
                        fontSize: 9.8,
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
              ),

              const Icon(
                Icons.chevron_right_rounded,
                color: Colors.white24,
                size: 20,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildLogoutButton() {
    return SizedBox(
      width: double.infinity,
      child: OutlinedButton.icon(
        onPressed: _showLogoutDialog,
        icon: const Icon(
          Icons.logout_rounded,
          size: 19,
        ),
        label: Text(
          'Logout',
          style: GoogleFonts.dmSans(
            fontWeight: FontWeight.w700,
          ),
        ),
        style: OutlinedButton.styleFrom(
          foregroundColor: red,
          padding: const EdgeInsets.symmetric(
            vertical: 15,
          ),
          side: BorderSide(
            color: red.withOpacity(.32),
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(17),
          ),
        ),
      ),
    );
  }

  Widget _glassCard({
    required Widget child,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(
        horizontal: 15,
        vertical: 3,
      ),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: mint.withOpacity(.10),
        ),
      ),
      child: child,
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      color: Colors.white.withOpacity(.06),
    );
  }
}

class _ProfileInfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final bool isLast;

  const _ProfileInfoRow({
    required this.icon,
    required this.label,
    required this.value,
    this.isLast = false,
  });

  @override
  Widget build(BuildContext context) {
    const mint = Color(0xFF85FFF2);

    return Container(
      padding: const EdgeInsets.symmetric(
        vertical: 13,
      ),
      decoration: BoxDecoration(
        border: isLast
            ? null
            : Border(
                bottom: BorderSide(
                  color: Colors.white.withOpacity(.06),
                ),
              ),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: mint.withOpacity(.07),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              color: mint,
              size: 19,
            ),
          ),

          const SizedBox(width: 11),

          Expanded(
            flex: 4,
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white54,
                fontSize: 11.3,
              ),
            ),
          ),

          const SizedBox(width: 8),

          Expanded(
            flex: 6,
            child: Text(
              value,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              textAlign: TextAlign.end,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 11.3,
                height: 1.3,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
